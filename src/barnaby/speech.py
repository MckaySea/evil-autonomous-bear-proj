"""Piper speech preparation and blocking playback for the shared audio worker."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave


VOICE_PATH = Path(__file__).resolve().parents[2] / "voices" / "en_US-ryan-medium.onnx"


def clean_for_speech(text: str) -> str:
    """Remove Gemini's nonspoken stage directions and formatting."""
    text = re.sub(r"\*[^*]*\*", " ", text)
    text = re.sub(r"(?m)^\s*[-#]+\s*", "", text)
    return re.sub(r"\s+", " ", text.replace("`", "")).strip()


class SpeechPlayer:
    def __init__(self, voice_path: Path = VOICE_PATH):
        self.voice_path = Path(voice_path)
        self._voice = None

    def _load_voice(self) -> None:
        if self._voice is not None:
            return
        if not self.voice_path.is_file():
            raise FileNotFoundError(f"Piper voice not found: {self.voice_path}")
        for executable in ("sox", "pw-play"):
            if shutil.which(executable) is None:
                raise RuntimeError(f"{executable} is missing. Install sox and pipewire-bin.")
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(self.voice_path))
        print("[Barnaby TTS] Piper voice loaded", flush=True)

    @staticmethod
    def _run_process(command, stop, timeout: float) -> bool:
        """Drain stderr and stop subprocesses promptly when Barnaby exits."""
        if stop.is_set():
            return False
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.monotonic() + timeout
        try:
            while not stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{command[0]} timed out")
                try:
                    _, errors = process.communicate(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    continue
                if process.returncode != 0:
                    raise RuntimeError(f"{command[0]} failed: {errors.decode(errors='replace').strip()}")
                return True
            return False
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def play(self, text: str, stop, is_fresh) -> bool:
        text = clean_for_speech(text)
        if not text or not is_fresh():
            return False
        self._load_voice()
        if not is_fresh():
            return False
        with tempfile.TemporaryDirectory(prefix="barnaby-tts-") as directory:
            normal_wav = Path(directory) / "piper.wav"
            demon_wav = Path(directory) / "demon.wav"
            with wave.open(str(normal_wav), "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)
            if not is_fresh():
                return False
            if not self._run_process(
                ["sox", str(normal_wav), str(demon_wav),
                 "pitch", "-650", "tempo", "0.90", "reverb", "30"],
                stop, 60,
            ):
                return False
            # API time, queue time, voice loading, synthesis and effects all count.
            if not is_fresh():
                return False
            return self._run_process(["pw-play", str(demon_wav)], stop, 120)


def speak(text: str, captured_at: float | None = None) -> None:
    """Compatibility entry point; speech uses the same queue as effects."""
    if os.environ.get("BARNABY_TTS") == "1":
        from .reactions import get_audio

        get_audio().submit_observation(
            clean_for_speech(text), time.monotonic() if captured_at is None else captured_at,
        )
