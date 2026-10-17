"""Audio scheduling policy and one playback worker; standard library only."""

from collections import deque
from dataclasses import dataclass
import math
from threading import Condition, Event, Thread
import time


@dataclass(frozen=True)
class AudioSettings:
    observation_expiry: float = 8.0
    effect_expiry: float = 5.0
    effect_cooldown: float = 5.0
    playback_gap: float = 1.0
    gesture_release: float = 0.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.observation_expiry == 0 or self.effect_expiry == 0:
            raise ValueError("Audio expiry times must be positive")


@dataclass(frozen=True)
class AudioItem:
    kind: str
    content: str
    created_at: float
    expires_at: float


class ReleaseGate:
    """Feed every inference, including unchanged labels and 'none'.

    Trigger once per occurrence; brief detection dropouts do not rearm it.
    A rejected trigger still disarms that occurrence (e.g. during cooldown).
    """

    def __init__(self, release_seconds: float):
        self.release_seconds = release_seconds
        self._disarmed: dict[str, float | None] = {}

    def update(self, label: str, now: float) -> bool:
        for previous, absent_since in list(self._disarmed.items()):
            if previous == label:
                if absent_since is not None and now - absent_since >= self.release_seconds:
                    del self._disarmed[previous]
                else:
                    self._disarmed[previous] = None
            elif absent_since is None:
                self._disarmed[previous] = now
            elif now - absent_since >= self.release_seconds:
                del self._disarmed[previous]
        if label in ("none", "unknown") or label in self._disarmed:
            return False
        self._disarmed[label] = None
        return True


class AudioSchedule:
    """Deterministic policy. Its owner supplies time and serializes access."""

    def __init__(self, settings: AudioSettings):
        self.settings = settings
        self.effects: deque[AudioItem] = deque()
        self.observation: AudioItem | None = None
        self.current: AudioItem | None = None
        self.cooldown_until: dict[str, float] = {}
        self.next_play_at = 0.0

    def prune(self, now: float) -> None:
        self.effects = deque(item for item in self.effects if item.expires_at > now)
        if self.observation is not None and self.observation.expires_at <= now:
            self.observation = None

    def add_effect(self, name: str, now: float) -> bool:
        self.prune(now)
        if now < self.cooldown_until.get(name, 0.0):
            return False
        if self.current is not None and self.current.kind == "effect" and self.current.content == name:
            return False
        if len(self.effects) >= 2 or any(item.content == name for item in self.effects):
            return False
        self.effects.append(AudioItem("effect", name, now, now + self.settings.effect_expiry))
        return True

    def add_observation(self, text: str, captured_at: float, now: float) -> bool:
        self.prune(now)
        expires_at = captured_at + self.settings.observation_expiry
        if not text.strip() or expires_at <= now:
            return False
        if self.observation is not None and captured_at < self.observation.created_at:
            return False
        self.observation = AudioItem("observation", text, captured_at, expires_at)
        return True

    def wants_observation(self, now: float) -> bool:
        self.prune(now)
        return self.observation is None and (
            self.current is None or self.current.kind != "observation"
        )

    def take_next(self, now: float) -> AudioItem | None:
        self.prune(now)
        if self.current is not None or now < self.next_play_at:
            return None
        # Fresh speech always goes before waiting effects; active audio finishes.
        if self.observation is not None:
            self.current, self.observation = self.observation, None
        elif self.effects:
            self.current = self.effects.popleft()
        return self.current

    def finish(self, now: float, played: bool) -> None:
        if self.current is not None and played:
            self.next_play_at = now + self.settings.playback_gap
            if self.current.kind == "effect":
                self.cooldown_until[self.current.content] = now + self.settings.effect_cooldown
        self.current = None


class AudioCoordinator:
    """One worker owns playback. The backend blocks until playback ends.

    backend.play(item, stop_event, is_fresh) returns whether audio played.
    It must check is_fresh immediately before playback, after preparation.
    """

    def __init__(self, backend, settings: AudioSettings = AudioSettings(), *, clock=time.monotonic):
        self.settings = settings
        self._backend = backend
        self._clock = clock
        self._schedule = AudioSchedule(settings)
        self._condition = Condition()
        self._stop = Event()
        self._thread = Thread(target=self._run, name="barnaby-audio", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit_effect(self, name: str) -> bool:
        with self._condition:
            if self._stop.is_set():
                return False
            accepted = self._schedule.add_effect(name, self._clock())
            self._condition.notify()
            return accepted

    def submit_observation(self, text: str, captured_at: float) -> bool:
        with self._condition:
            if self._stop.is_set():
                return False
            accepted = self._schedule.add_observation(text, captured_at, self._clock())
            self._condition.notify()
            return accepted

    def wants_observation(self) -> bool:
        with self._condition:
            return not self._stop.is_set() and self._schedule.wants_observation(self._clock())

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                with self._condition:
                    item = self._schedule.take_next(self._clock())
                    while item is None and not self._stop.is_set():
                        delay = max(0.0, self._schedule.next_play_at - self._clock())
                        self._condition.wait(timeout=delay if delay > 0 else None)
                        item = self._schedule.take_next(self._clock())
                    if self._stop.is_set():
                        break
                fresh = lambda: not self._stop.is_set() and self._clock() < item.expires_at
                played = False
                try:
                    if fresh():
                        played = self._backend.play(item, self._stop, fresh)
                    if not played and not fresh() and not self._stop.is_set():
                        print(f"[Barnaby audio] Dropped expired {item.kind}", flush=True)
                except Exception as error:
                    # Playback may have partially succeeded; leave a gap on failure.
                    played = True
                    if not self._stop.is_set():
                        print(f"[Barnaby audio] {error}", flush=True)
                finally:
                    with self._condition:
                        self._schedule.finish(self._clock(), played)
        finally:
            self._backend.close()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._schedule.effects.clear()
            self._schedule.observation = None
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
