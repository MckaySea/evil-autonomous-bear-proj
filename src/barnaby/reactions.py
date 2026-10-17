"""Route perception and Gemini events through one audio coordinator."""

import atexit
from dataclasses import dataclass
import json
import os
from threading import Lock
import time

from .audio_coordinator import AudioCoordinator, AudioSettings, ReleaseGate
from .sounds import EFFECTS
from .speech import clean_for_speech


class AudioPlayback:
    """Load hardware dependencies only inside the audio worker, when needed."""

    def __init__(self):
        self.effects = None
        self.speech = None

    def play(self, item, stop, is_fresh) -> bool:
        if item.kind == "effect":
            if self.effects is None:
                from .sounds import BarnabyAudio

                self.effects = BarnabyAudio()
            return self.effects.play(item.content, stop, is_fresh)
        if self.speech is None:
            from .speech import SpeechPlayer

            self.speech = SpeechPlayer()
        return self.speech.play(item.content, stop, is_fresh)

    def close(self) -> None:
        if self.effects is not None:
            self.effects.close()


_audio: AudioCoordinator | None = None
_audio_lock = Lock()
_settings = AudioSettings()
_gesture_gate = ReleaseGate(_settings.gesture_release)
_expression_gate = ReleaseGate(_settings.gesture_release)
_previous = ("none", "none")


def configure_audio(settings: AudioSettings) -> None:
    global _settings, _gesture_gate, _expression_gate, _previous
    stop_audio()
    _settings = settings
    _gesture_gate = ReleaseGate(settings.gesture_release)
    _expression_gate = ReleaseGate(settings.gesture_release)
    _previous = ("none", "none")


def get_audio() -> AudioCoordinator:
    global _audio
    with _audio_lock:
        if _audio is None:
            _audio = AudioCoordinator(AudioPlayback(), _settings)
            _audio.start()
        return _audio


def stop_audio() -> None:
    global _audio
    with _audio_lock:
        if _audio is not None:
            _audio.stop()
            _audio = None


atexit.register(stop_audio)


@dataclass(frozen=True)
class PerceptionEvent:
    gesture: str
    expression: str
    hand_confidence: float
    face_confidence: float
    inference_ms: float


def on_perception(event: PerceptionEvent) -> None:
    """Called every inference so release timing includes unchanged samples."""
    global _previous
    labels = (event.gesture, event.expression)
    if labels != _previous:
        print(json.dumps(event.__dict__), flush=True)
        _previous = labels
    now = time.monotonic()
    for gate, label in ((_gesture_gate, event.gesture), (_expression_gate, event.expression)):
        if gate.update(label, now) and label in EFFECTS:
            get_audio().submit_effect(label)


def wants_scene_description() -> bool:
    # Text-only Gemini use has no speech backlog to wait for.
    return os.environ.get("BARNABY_TTS") != "1" or get_audio().wants_observation()


def on_scene_description(description: str, captured_at: float) -> None:
    if time.monotonic() >= captured_at + _settings.observation_expiry:
        print("[Barnaby sees] Dropped expired observation", flush=True)
        return
    print(f"[Barnaby sees] {description}", flush=True)
    if os.environ.get("BARNABY_TTS") == "1":
        get_audio().submit_observation(clean_for_speech(description), captured_at)
