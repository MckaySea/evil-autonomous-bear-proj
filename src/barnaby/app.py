"""Camera demo: python -m barnaby --source 0 (press Q to quit)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import time

import cv2

from .camera import Camera
from .audio_coordinator import AudioSettings
from .face_memory import FaceMemory, DEFAULT_THRESHOLD
from .gestures import HAND_CONNECTIONS, StableLabel
from .models import DEFAULT_MODEL_DIR
from .reactions import (
    PerceptionEvent, configure_audio, on_perception, on_scene_description,
    stop_audio, wants_scene_description,
)
from .scene import MODEL, describe_scene
from .vision import Observation, Vision
from dotenv import load_dotenv

def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def overlay(frame, observation: Observation, gesture: str, expression: str, person=None):
    if observation.face_box:
        x, y, w, h = observation.face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
    if observation.hand_points is not None:
        points = observation.hand_points[:, :2].astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(points[a]), tuple(points[b]), (255, 160, 0), 2)
        for point in points:
            cv2.circle(frame, tuple(point), 3, (0, 240, 255), -1)
    text = f"Gesture: {gesture} | Expression: {expression} | {observation.inference_ms:.0f} ms"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1, cv2.LINE_AA)
    if person:
        cv2.putText(frame, f"Recognized: {person}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2, cv2.LINE_AA)


def main(argv=None) -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="USB camera index or video path")
    parser.add_argument("--picamera", action="store_true", help="Use a Pi CSI camera via Picamera2")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--width", type=positive, default=640)
    parser.add_argument("--height", type=positive, default=480)
    parser.add_argument("--inference-width", type=positive, default=320, help="Downscale inference to save CPU")
    parser.add_argument("--interval", type=float, default=0.15, help="Minimum seconds between inference starts")
    parser.add_argument("--threads", type=positive, default=2)
    parser.add_argument("--faces", action="store_true", help="Recognize enrolled people locally")
    parser.add_argument("--face-interval", type=float, default=1.5,
                        help="Seconds between SFace recognition attempts")
    parser.add_argument("--face-threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Minimum cosine similarity for a known person")
    parser.add_argument("--stable-frames", type=positive, default=3, help="Consecutive inference observations")
    parser.add_argument("--headless", action="store_true", help="Print events without a preview window")
    parser.add_argument("--max-frames", type=positive, help="Stop after this many captured frames")
    parser.add_argument("--realtime", action="store_true", help="Play a video at its source frame rate")
    parser.add_argument("--loop", action="store_true", help="Repeat a video until Q or Ctrl+C")
    parser.add_argument("--save-video", type=Path, help="Write an annotated MP4, including in headless mode")
    parser.add_argument("--check", action="store_true", help="Verify weights and run all models without a camera")

    # Optional cloud captioning; the local detection models continue as before.
    parser.add_argument("--scene", action="store_true", help="Describe people using Gemini 3.8 Flash")
    parser.add_argument("--scene-model", default=MODEL, help="Gemini model ID")
    parser.add_argument("--scene-interval", type=float, default=10.0,
                        help="Minimum seconds between starting Gemini requests")
    parser.add_argument("--scene-timeout", type=float, default=25.0,
                        help="Gemini request timeout in seconds")
    parser.add_argument("--scene-image-width", type=positive, default=512,
                        help="Maximum width of image submitted to Gemini")
    parser.add_argument("--observation-expiry", type=float, default=8.0,
                        help="Maximum observation age at speech start, from image capture (seconds)")
    parser.add_argument("--effect-expiry", type=float, default=5.0,
                        help="Discard sound effects waiting this many seconds")
    parser.add_argument("--effect-cooldown", type=float, default=5.0,
                        help="Seconds after an effect finishes before it can trigger again")
    parser.add_argument("--audio-gap", type=float, default=1.0,
                        help="Minimum silence between audio items (seconds)")
    parser.add_argument("--gesture-release", type=float, default=0.5,
                        help="Seconds a gesture must be absent before it can trigger again")
    args = parser.parse_args(argv)

    try:
        audio_settings = AudioSettings(
            observation_expiry=args.observation_expiry,
            effect_expiry=args.effect_expiry,
            effect_cooldown=args.effect_cooldown,
            playback_gap=args.audio_gap,
            gesture_release=args.gesture_release,
        )
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error("--interval must be finite and nonnegative")
    if not math.isfinite(args.scene_interval) or args.scene_interval <= 0:
        parser.error("--scene-interval must be finite and positive")
    if not math.isfinite(args.scene_timeout) or args.scene_timeout <= 0:
        parser.error("--scene-timeout must be finite and positive")
    if not math.isfinite(args.face_interval) or args.face_interval <= 0:
        parser.error("--face-interval must be finite and positive")
    if not math.isfinite(args.face_threshold) or not -1 <= args.face_threshold <= 1:
        parser.error("--face-threshold must be between -1 and 1")
    if args.scene and not args.check and not os.getenv("GEMINI_API_KEY"):
        parser.error("GEMINI_API_KEY must be set when --scene is enabled")
    if args.save_video and not args.picamera and not args.source.isdecimal():
        if args.save_video.resolve() == Path(args.source).resolve():
            parser.error("--save-video must differ from the input video")

    camera = None
    writer = None
    scene_executor = None
    scene_future = None
    scene_captured_at = 0.0
    next_scene_at = 0.0
    face_memory = None
    next_face_at = 0.0
    current_person = None
    candidate_person = None
    candidate_hits = 0
    try:
        vision = Vision(args.model_dir, args.threads)
        if args.check:
            vision.check()
            print(f"All four models verified and ran on CPU (OpenCV {cv2.__version__}).")
            return

        configure_audio(audio_settings)
        if args.faces:
            face_memory = FaceMemory(args.model_dir)
            print("Local SFace recognition enabled; requires two matching checks")
        camera = Camera(args.source, args.width, args.height, args.picamera, args.realtime, args.loop)
        if args.scene:
            scene_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="barnaby-gemini")

        gesture_filter, expression_filter = StableLabel(args.stable_frames), StableLabel(args.stable_frames)
        count, next_inference = 0, 0.0
        observation = Observation()
        print("Barnaby vision running. Press Q in the preview or Ctrl+C to stop.")
        if args.scene:
            print(f"Gemini descriptions enabled: {args.scene_model} (every {args.scene_interval:g}s)")

        while True:
            ok, frame = camera.read()
            frame_captured_at = time.monotonic()
            if not ok:
                if camera.is_file:
                    break
                raise RuntimeError("Camera stopped returning frames")
            if camera.looped:
                gesture_filter = StableLabel(args.stable_frames)
                expression_filter = StableLabel(args.stable_frames)
                observation, next_inference = Observation(), 0.0

            scale = min(1.0, args.inference_width / frame.shape[1])
            small = cv2.resize(
                frame,
                (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
            )
            now = time.monotonic()

            # Offline analysis visits every frame; realtime playback uses the camera interval.
            if (camera.is_file and not args.realtime) or now >= next_inference:
                observation = vision.infer(small)
                next_inference = now + args.interval
                labels = (gesture_filter.update(observation.gesture), expression_filter.update(observation.expression))
                # Release gates need every sample, not just label changes.
                on_perception(PerceptionEvent(*labels, observation.hand_confidence,
                                               observation.face_confidence, observation.inference_ms))

            if face_memory is not None and now >= next_face_at:
                next_face_at = now + args.face_interval
                # Run on the full-resolution frame, not the smaller hand-inference image.
                person, similarity = face_memory.recognize(frame, args.face_threshold)
                if person == candidate_person:
                    candidate_hits += 1
                else:
                    candidate_person, candidate_hits = person, 1
                if candidate_hits >= 2 and current_person != candidate_person:
                    current_person = candidate_person
                    if current_person:
                        print(f"[Barnaby recognizes] {current_person} "
                              f"(similarity {similarity:.3f})", flush=True)

            if scene_executor is not None:
                # Consume finished Gemini responses without waiting for the network.
                if scene_future is not None and scene_future.done():
                    try:
                        description = scene_future.result()
                        on_scene_description(description, scene_captured_at)
                    except Exception as error:
                        print(f"[Barnaby Gemini error] {error}", flush=True)
                        # Back off after failed requests (e.g., rate limits).
                        next_scene_at = max(next_scene_at, time.monotonic() + 30.0)
                    finally:
                        scene_future = None

                # Sample the original camera frame so people remain visible.
                # Never queue up overlapping requests if the API is slow.
                if (scene_future is None and time.monotonic() >= next_scene_at
                        and wants_scene_description()):
                    scene_captured_at = frame_captured_at
                    scene_future = scene_executor.submit(
                        describe_scene,
                        frame.copy(),
                        args.scene_model,
                        image_width=args.scene_image_width,
                        timeout=args.scene_timeout,
                        recognized_name=current_person,
                    )
                    next_scene_at = time.monotonic() + args.scene_interval

            if not args.headless or args.save_video:
                overlay(small, observation, gesture_filter.value,
                        expression_filter.value, current_person)
            if args.save_video:
                if writer is None:
                    args.save_video.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                             camera.fps, (small.shape[1], small.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create output video {args.save_video}")
                writer.write(small)
            if not args.headless:
                cv2.imshow("Barnaby Bear", small)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            count += 1
            if args.max_frames and count >= args.max_frames:
                break
            if args.headless and not camera.is_file:
                time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Barnaby could not start/run: {error}\n")
    finally:
        stop_audio()
        if scene_executor is not None:
            scene_executor.shutdown(wait=False, cancel_futures=True)
        if face_memory is not None:
            face_memory.close()
        if writer is not None:
            writer.release()
        if camera is not None:
            camera.close()
        if not args.headless and not args.check:
            cv2.destroyAllWindows()
