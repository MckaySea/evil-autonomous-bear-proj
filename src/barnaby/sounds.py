from pathlib import Path
import time
import pygame

ROOT = Path(__file__).resolve().parents[2]
SOUND_DIR = ROOT / "sounds"

EFFECTS = {
    "middle_finger": "growl.wav",
    "peace": "sexy.wav"
}


class BarnabyAudio:
    def __init__(self):
        pygame.mixer.init(
            frequency=44100,
            size=-16,
            channels=2,
            buffer=512,
        )

        self.effects = {}
        self.last_played = {}
        self.cooldown = 5.0

        for event, filename in EFFECTS.items():
            path = SOUND_DIR / "effects" / filename

            if path.exists():
                self.effects[event] = pygame.mixer.Sound(str(path))
                self.effects[event].set_volume(0.6)

    def play(self, event):
        if event not in self.effects:
            return

        now = time.monotonic()

        if now - self.last_played.get(event, -float("inf")) < self.cooldown:
            return

        self.effects[event].play()
        self.last_played[event] = now



    def stop_music(self):
        pygame.mixer.music.stop()
