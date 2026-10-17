"""Blocking sound-effect playback, called only by the shared audio worker."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOUND_DIR = ROOT / "sounds"

EFFECTS = {
    "middle_finger": "growl.wav",
    "peace": "sexy.wav",
}


class BarnabyAudio:
    def __init__(self):
        import pygame

        self._mixer = pygame.mixer
        self._mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.effects = {}

    def play(self, event, stop, is_fresh) -> bool:
        if event not in EFFECTS or stop.is_set():
            return False
        if event not in self.effects:
            path = SOUND_DIR / "effects" / EFFECTS[event]
            if not path.is_file():
                raise FileNotFoundError(f"Sound effect not found: {path}")
            sound = self._mixer.Sound(str(path))
            sound.set_volume(0.6)
            self.effects[event] = sound
        if not is_fresh():
            return False
        channel = self.effects[event].play()
        if channel is None:
            raise RuntimeError("No audio channel available for sound effect")
        try:
            while channel.get_busy():
                if stop.wait(0.02):
                    break
        finally:
            if stop.is_set():
                channel.stop()
        return True

    def close(self) -> None:
        self._mixer.quit()
