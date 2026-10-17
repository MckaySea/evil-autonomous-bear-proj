"""Apply minimal, verified face-memory integration to an existing Barnaby checkout.

Usage from the repo root after copying this file and face_memory.py:
    python install_face_memory.py

Makes backups before writing. Aborts without writing if upstream anchors differ.
"""
from pathlib import Path
import shutil

ROOT = Path.cwd()
APP = ROOT / "src" / "barnaby" / "app.py"
SCENE = ROOT / "src" / "barnaby" / "scene.py"
MOD = ROOT / "src" / "barnaby" / "face_memory.py"


def swap(text, old, new, label):
    if old not in text:
        raise RuntimeError(f"{label}: could not find expected code. Nothing changed.")
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected unique code but found {text.count(old)} copies")
    return text.replace(old, new, 1)


def main():
    if not all(x.exists() for x in (APP, SCENE, MOD)):
        raise RuntimeError("Run from the repository root; copy face_memory.py into src/barnaby first")
    app, scene = APP.read_text(), SCENE.read_text()
    if 'from .face_memory import FaceMemory' in app:
        print("Face memory integration already present. No files changed.")
        return

    app = swap(app, "from .camera import Camera\n",
        "from .camera import Camera\nfrom .face_memory import FaceMemory, DEFAULT_THRESHOLD\n", "app imports")
    app = swap(app,
        "def overlay(frame, observation: Observation, gesture: str, expression: str):",
        "def overlay(frame, observation: Observation, gesture: str, expression: str, person=None):",
        "overlay signature")
    app = swap(app,
        '    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1, cv2.LINE_AA)\n',
        '    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1, cv2.LINE_AA)\n'
        '    if person:\n'
        '        cv2.putText(frame, f"Recognized: {person}", (10, 50),\n'
        '                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2, cv2.LINE_AA)\n',
        "overlay rendering")
    app = swap(app,
        '    parser.add_argument("--threads", type=positive, default=2)\n',
        '    parser.add_argument("--threads", type=positive, default=2)\n'
        '    parser.add_argument("--faces", action="store_true", help="Recognize enrolled people locally")\n'
        '    parser.add_argument("--face-interval", type=float, default=1.5,\n'
        '                        help="Seconds between SFace recognition attempts")\n'
        '    parser.add_argument("--face-threshold", type=float, default=DEFAULT_THRESHOLD,\n'
        '                        help="Minimum cosine similarity for a known person")\n',
        "face arguments")
    app = swap(app,
        '    if args.scene and not args.check and not os.getenv("GEMINI_API_KEY"):\n',
        '    if not math.isfinite(args.face_interval) or args.face_interval <= 0:\n'
        '        parser.error("--face-interval must be finite and positive")\n'
        '    if not math.isfinite(args.face_threshold) or not -1 <= args.face_threshold <= 1:\n'
        '        parser.error("--face-threshold must be between -1 and 1")\n'
        '    if args.scene and not args.check and not os.getenv("GEMINI_API_KEY"):\n',
        "face argument validation")
    app = swap(app,
        '    next_scene_at = 0.0\n',
        '    next_scene_at = 0.0\n'
        '    face_memory = None\n'
        '    next_face_at = 0.0\n'
        '    current_person = None\n'
        '    candidate_person = None\n'
        '    candidate_hits = 0\n',
        "face state")
    app = swap(app,
        '        camera = Camera(args.source, args.width, args.height, args.picamera, args.realtime, args.loop)\n',
        '        if args.faces:\n'
        '            face_memory = FaceMemory(args.model_dir)\n'
        '            print("Local SFace recognition enabled; requires two matching checks")\n'
        '        camera = Camera(args.source, args.width, args.height, args.picamera, args.realtime, args.loop)\n',
        "face initialization")
    app = swap(app,
        '            if scene_executor is not None:\n',
        '            if face_memory is not None and now >= next_face_at:\n'
        '                next_face_at = now + args.face_interval\n'
        '                # Run on the full-resolution frame, not the smaller hand-inference image.\n'
        '                person, similarity = face_memory.recognize(frame, args.face_threshold)\n'
        '                if person == candidate_person:\n'
        '                    candidate_hits += 1\n'
        '                else:\n'
        '                    candidate_person, candidate_hits = person, 1\n'
        '                if candidate_hits >= 2 and current_person != candidate_person:\n'
        '                    current_person = candidate_person\n'
        '                    if current_person:\n'
        '                        print(f"[Barnaby recognizes] {current_person} "\n'
        '                              f"(similarity {similarity:.3f})", flush=True)\n'
        '\n'
        '            if scene_executor is not None:\n',
        "face recognition scheduling")
    app = swap(app,
        '                        timeout=args.scene_timeout,\n',
        '                        timeout=args.scene_timeout,\n'
        '                        recognized_name=current_person,\n',
        "Gemini name forwarding")
    app = swap(app,
        '                overlay(small, observation, gesture_filter.value, expression_filter.value)\n',
        '                overlay(small, observation, gesture_filter.value,\n'
        '                        expression_filter.value, current_person)\n',
        "identity overlay")
    app = swap(app,
        '        if writer is not None:\n            writer.release()\n',
        '        if face_memory is not None:\n            face_memory.close()\n'
        '        if writer is not None:\n            writer.release()\n',
        "face db cleanup")

    scene = swap(scene,
        '    timeout: float = 25.0,\n',
        '    timeout: float = 25.0,\n    recognized_name: str | None = None,\n',
        "scene parameter")
    scene = swap(scene,
        '    try:\n        response = client.interactions.create(\n',
        '    prompt = PROMPT\n'
        '    if recognized_name:\n'
        '        prompt += (\n'
        '            " The visitor voluntarily enrolled and their recognized name is "\n'
        '            + repr(recognized_name) + ". Address them by name naturally. "\n'
        '            "Do not invent other personal details."\n'
        '        )\n'
        '    try:\n        response = client.interactions.create(\n',
        "Gemini prompt prep")
    scene = swap(scene,
        '                {"type": "text", "text": PROMPT},\n',
        '                {"type": "text", "text": prompt},\n',
        "Gemini prompt")

    # Compile both candidates before changing user files.
    compile(app, str(APP), "exec")
    compile(scene, str(SCENE), "exec")
    for path, content in ((APP, app), (SCENE, scene)):
        backup = path.with_suffix(path.suffix + ".pre-face-memory.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(content)
        print(f"Updated {path}; backup {backup}")
    print("Face memory integrated. Run python -m barnaby --faces --scene ...")


if __name__ == "__main__":
    main()
