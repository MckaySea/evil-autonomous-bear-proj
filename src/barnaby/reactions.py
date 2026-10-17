"""Attach Barnaby's speech, LEDs, or servo commands here."""

from dataclasses import dataclass
import json
from .sounds import BarnabyAudio

from .speech import speak


audio = BarnabyAudio()

@dataclass(frozen=True)
class PerceptionEvent:
    gesture: str
    expression: str
    hand_confidence: float
    face_confidence: float
    inference_ms: float


def on_perception(event: PerceptionEvent) -> None:
    print(json.dumps(event.__dict__), flush=True)

    audio.play(event.gesture)
    audio.play(event.expression)
    



def on_scene_description(description: str) -> None:
    """Print Gemini's response and have Barnaby speak it."""
    print(f"[Barnaby sees] {description}", flush=True)
    speak(description)
