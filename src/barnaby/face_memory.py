"""Opt-in face enrollment and recognition with OpenCV YuNet + SFace.

Run from the repository root:
  python -m barnaby.face_memory enroll --name "Alex" --source 0
  python -m barnaby.face_memory enroll --name "Sam" --images photos/sam1.jpg photos/sam2.jpg
  python -m barnaby.face_memory test --source 0
  python -m barnaby.face_memory list
  python -m barnaby.face_memory delete --name "Alex"
"""

import argparse
import os
from pathlib import Path
import sqlite3
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "data" / "faces.db"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
DEFAULT_THRESHOLD = 0.50  # More conservative than OpenCV's example 0.363.


class FaceMemory:
    def __init__(self, model_dir: Path = MODEL_DIR, db_path: Path = DB_PATH):
        model_dir = Path(model_dir)
        sface = model_dir / SFACE_FILENAME
        yunet = model_dir / YUNET_FILENAME
        for model in (sface, yunet):
            if not model.is_file():
                raise FileNotFoundError(f"Missing model: {model}")

        self.detector = cv2.FaceDetectorYN.create(
            str(yunet), "", (320, 240), 0.85, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # A database created here is readable only by the owning Unix account.
        if not db_path.exists():
            descriptor = os.open(db_path, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(descriptor)
        os.chmod(db_path, 0o600)
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("""CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE
        )""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
        )""")
        self.db.commit()
        self._refresh_cache()

    def _refresh_cache(self):
        rows = self.db.execute(
            """SELECT people.name, samples.embedding FROM samples
               JOIN people ON samples.person_id = people.id"""
        ).fetchall()
        self.known = [(name, np.frombuffer(blob, dtype=np.float32))
                      for name, blob in rows]

    def close(self):
        self.db.close()

    def embedding(self, frame: np.ndarray) -> np.ndarray | None:
        """Return a normalized face feature only when exactly one good face is found."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) != 1:
            return None
        face = faces[0]
        if min(face[2], face[3]) < 75:  # Avoid tiny, unreliable face crops.
            return None
        aligned = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned).astype(np.float32).ravel()
        magnitude = np.linalg.norm(feature)
        if not np.isfinite(magnitude) or magnitude == 0:
            return None
        return feature / magnitude

    def recognize(self, frame: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
        """Return (name_or_None, best_cosine_score); unknown faces stay unknown."""
        query = self.embedding(frame)
        if query is None or not self.known:
            return None, None
        best_name, best_score = None, -1.0
        for name, stored in self.known:
            if stored.shape != query.shape:
                continue
            score = float(np.dot(query, stored))
            if score > best_score:
                best_name, best_score = name, score
        return (best_name if best_score >= threshold else None), best_score

    def enroll(self, name: str, features: list[np.ndarray]):
        name = name.strip()
        if not name or len(name) > 50:
            raise ValueError("Name must be between 1 and 50 characters")
        if not features:
            raise ValueError("No valid face samples collected")
        # Replace old samples for this name rather than silently accumulating them.
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO people(name) VALUES (?)", (name,))
            person_id = self.db.execute(
                "SELECT id FROM people WHERE name = ?", (name,)
            ).fetchone()[0]
            self.db.execute("DELETE FROM samples WHERE person_id = ?", (person_id,))
            self.db.executemany(
                "INSERT INTO samples(person_id, embedding) VALUES (?, ?)",
                [(person_id, feature.astype(np.float32).tobytes())
                 for feature in features],
            )
        self._refresh_cache()
        print(f"Saved {len(features)} samples for {name}")

    def list_people(self):
        rows = self.db.execute("""SELECT people.name, COUNT(samples.id)
            FROM people LEFT JOIN samples ON samples.person_id = people.id
            GROUP BY people.id ORDER BY people.name""").fetchall()
        for name, samples in rows:
            print(f"{name}: {samples} face samples")
        if not rows:
            print("No faces enrolled")

    def delete(self, name: str):
        with self.db:
            cursor = self.db.execute("DELETE FROM people WHERE name = ?", (name,))
        self._refresh_cache()
        print("Deleted" if cursor.rowcount else "Not found", name)


def open_camera(source: str):
    camera = cv2.VideoCapture(int(source) if source.isdecimal() else source)
    if not camera.isOpened():
        raise RuntimeError(f"Camera source {source!r} could not be opened")
    return camera


def enroll_camera(memory: FaceMemory, name: str, source: str, count: int):
    camera = open_camera(source)
    samples = []
    try:
        print("Stand in good light, with ONLY ONE face visible.")
        print("Press SPACE to capture each sample; change your head angle; Q to quit.")
        while len(samples) < count:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            preview = frame.copy()
            cv2.putText(preview, f"{name}: {len(samples)}/{count}  SPACE=capture Q=quit",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("Barnaby face enrollment", preview)
            key = cv2.waitKey(1) & 0xff
            if key == ord("q"):
                break
            if key == 32:
                feature = memory.embedding(frame)
                if feature is None:
                    print("No usable face: need exactly one clear face, at least 75px tall.")
                else:
                    samples.append(feature)
                    print(f"Captured sample {len(samples)}/{count}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    if len(samples) == count:
        memory.enroll(name, samples)
    else:
        print("Canceled; no new samples saved")


def enroll_images(memory: FaceMemory, name: str, paths: list[str]):
    features = []
    for path in paths:
        frame = cv2.imread(path)
        feature = memory.embedding(frame)
        if feature is None:
            print(f"Skipped {path}: need exactly one clear face")
        else:
            features.append(feature)
    if features:
        memory.enroll(name, features)
    else:
        print("No usable images; nothing saved")


def test_camera(memory: FaceMemory, source: str, threshold: float):
    camera = open_camera(source)
    last, label, score = 0.0, "Unknown", None
    try:
        print("Hold a face in view; Q quits. Scores are cosine similarity.")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            if time.monotonic() - last >= 1.0:
                name, score = memory.recognize(frame, threshold)
                label = name or "Unknown"
                last = time.monotonic()
            text = f"{label} ({score:.3f})" if score is not None else label
            cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.imshow("Barnaby recognition test", frame)
            if cv2.waitKey(1) & 0xff == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    sub = parser.add_subparsers(dest="cmd", required=True)
    enroll = sub.add_parser("enroll", help="Enroll one consenting person's face")
    enroll.add_argument("--name", required=True)
    inputs = enroll.add_mutually_exclusive_group()
    inputs.add_argument("--source", default="0")
    inputs.add_argument("--images", nargs="+", metavar="IMAGE")
    enroll.add_argument("--samples", type=int, default=5)
    test = sub.add_parser("test", help="Test face matching on a live webcam")
    test.add_argument("--source", default="0")
    test.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    sub.add_parser("list")
    delete = sub.add_parser("delete")
    delete.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.cmd == "enroll" and args.samples < 1:
        parser.error("--samples must be positive")
    if args.cmd == "test" and not -1 <= args.threshold <= 1:
        parser.error("--threshold must be between -1 and 1")
    memory = FaceMemory(args.model_dir, args.db)
    try:
        if args.cmd == "enroll":
            if args.images:
                enroll_images(memory, args.name, args.images)
            else:
                enroll_camera(memory, args.name, args.source, args.samples)
        elif args.cmd == "test":
            test_camera(memory, args.source, args.threshold)
        elif args.cmd == "list":
            memory.list_people()
        elif args.cmd == "delete":
            memory.delete(args.name)
    finally:
        memory.close()


if __name__ == "__main__":
    main()
