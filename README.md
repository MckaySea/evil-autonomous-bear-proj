# Barnaby Bear — Setup & Development Guide

Barnaby is a Halloween animatronic project built with Python and OpenCV. It detects hand gestures and facial expressions locally, recognizes previously enrolled faces with SFace, generates creepy/funny commentary with Gemini, speaks via Piper + SoX, and plays prerecorded sounds with `pygame.mixer`.

**Primary target:** Raspberry Pi 5 (64-bit Raspberry Pi OS). The USB webcam path also works on a Linux desktop with a compatible Python/OpenCV setup. The full audio stack currently assumes Linux + PipeWire.

## What runs where

| Component | Implementation | Runs on |
| --- | --- | --- |
| Palm and hand detection | OpenCV Zoo / MediaPipe ONNX models | Local CPU |
| Gesture classification | Hand-landmark rules in `gestures.py` | Local CPU |
| Face detection | YuNet | Local CPU |
| Facial expression classification | MobileFaceNet FER | Local CPU |
| Known-person recognition | SFace + SQLite (`data/faces.db`) | Local CPU |
| Scene descriptions and personalized dialogue | Gemini API | Cloud; requires internet/key |
| Spoken dialogue | Piper Ryan Medium + SoX demon effects | Local CPU |
| Prerecorded reactions | `pygame.mixer` | Local audio device |

## 1. Clone and install (Pi 5 / Debian / Ubuntu Linux)

Use a **64-bit OS**, Python **3.10–3.13**, an internet connection for initial setup, and a USB webcam. From a terminal:

```bash
sudo apt update
sudo apt install -y \
  git curl python3-venv python3-pip \
  libgl1 libglib2.0-0 v4l-utils \
  sox libsox-fmt-all alsa-utils pipewire-bin \
  libheif-examples

git clone https://github.com/MckaySea/evil-autonomous-bear-proj.git
cd evil-autonomous-bear-proj

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip install piper-tts google-genai python-dotenv pygame
```

**Important:** `requirements.txt` currently installs the core vision dependencies only. The full Pi setup also needs `pygame`, `piper-tts`, `google-genai` and `python-dotenv`. Audio libraries and devices initialize only when playback is needed; importing reactions or running `--check` does not initialize pygame. The queue and cooldown logic uses only Python's standard library.

Always run Barnaby commands from the **repository root**, where `models/`, `sounds/` and `.env` reside. Activate `.venv` in every new terminal:

```bash
cd /path/to/evil-autonomous-bear-proj
source .venv/bin/activate
```

### Windows or macOS development

A Linux machine is easiest for the **complete** camera + Piper/SoX + PipeWire setup. On Windows use `py -3.12 -m venv .venv` and `.venv\Scripts\activate` (PowerShell: `.venv\Scripts\Activate.ps1`); on macOS use `python3 -m venv .venv` and `source .venv/bin/activate`. Install the Python requirements as above. Camera detection and face enrollment may work, but **`speech.py` currently calls Linux's `sox` and `pw-play`**, so spoken audio needs a platform-specific playback replacement on Windows/macOS. The camera and Linux Pi steps below are the supported project workflow.

## 2. Check the models

The four baseline vision ONNX models and the Piper Ryan voice files are currently in GitHub. Check them:

```bash
ls -lh models/
ls -lh voices/
python -m barnaby --check
```

Expected voice files:

```text
voices/en_US-ryan-medium.onnx
voices/en_US-ryan-medium.onnx.json
```

If a baseline vision model is missing or corrupted, download and verify it using the pinned manifest:

```bash
python -m barnaby.models
```

If the Piper voice files are missing:

```bash
python -m piper.download_voices --data-dir voices en_US-ryan-medium
```

### Install SFace for recognizing people

**The SFace ONNX model is not tracked in Git**, so a fresh clone needs this additional download:

```bash
mkdir -p models
curl -fL --retry 3 \
  -o models/face_recognition_sface_2021dec.onnx \
  'https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'

python - <<'PY'
from pathlib import Path
import cv2
path = Path('models/face_recognition_sface_2021dec.onnx')
recognizer = cv2.FaceRecognizerSF.create(str(path), '')
print(f'SFace loaded successfully ({path.stat().st_size / 1e6:.1f} MB)')
PY
```

The `--check` command verifies only the four baseline networks, not SFace, and does not initialize audio.

## 3. Set up Gemini and Piper speech

Create a local `.env` file at the **project root**:

```bash
nano .env
```

Contents:

```dotenv
GEMINI_API_KEY=YOUR_GOOGLE_AI_STUDIO_KEY
BARNABY_TTS=1
```

Save, then restrict access:

```bash
chmod 600 .env
```

`src/barnaby/app.py` loads this file with `python-dotenv` automatically. `GEMINI_API_KEY` is required only when using `--scene`. `BARNABY_TTS=1` enables voice playback for Gemini descriptions. Without `--scene`, the local vision and face recognition can still run.

Current voice effects in `src/barnaby/speech.py`: **SoX `pitch -650 tempo 0.90 reverb 30`**. Playback uses the `pw-play` executable (with a hyphen, not `pwplay`).

Test Piper without waiting for Gemini:

```bash
BARNABY_TTS=1 python - <<'PY'
from barnaby.speech import speak
import time
speak('Hello, little human. Barnaby has awakened.')
time.sleep(20)
PY
```

You should see `[Barnaby TTS] Piper voice loaded` and hear audio. If no sound plays, confirm `command -v sox` and `command -v pw-play`, then check the audio output below.

## 4. Select the camera

For a USB webcam:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
python -m barnaby --source 0
```

Try `--source 1`, `--source 2`, etc. if the camera is on another device. A single camera may expose several `/dev/video*` nodes.

Press **Q** in Barnaby's preview window to stop, or **Ctrl+C** in the terminal. Use `--headless` when no GUI preview is needed.

### Optional: Raspberry Pi CSI/ribbon camera

The current repository provides `--picamera`, but Picamera2 needs Raspberry Pi OS camera packages. Use a **separate environment** so the system's libcamera bindings remain visible:

```bash
sudo apt install -y python3-picamera2 python3-opencv python3-numpy
python3 -m venv --system-site-packages .venv-camera
source .venv-camera/bin/activate
python -m pip install --no-deps -e .
python -m pip install piper-tts google-genai python-dotenv pygame
python -c 'import cv2; print(cv2.__version__)'
python -m barnaby --picamera
```

**Requirement:** `vision.py` checks for OpenCV **4.10 or newer**. Some Raspberry Pi OS releases ship an older `python3-opencv`; if the version check fails, the CSI environment needs an updated compatible OpenCV setup before it will run. The USB webcam setup is simpler. The standalone `face_memory enroll/test` CLI currently uses `cv2.VideoCapture` for live cameras; for a CSI camera, enroll from photos or adapt that CLI to Picamera2.

## 5. Set up sound effects and Bluetooth/HDMI audio

The repository currently has:

```text
sounds/effects/growl.wav
sounds/effects/sexy.wav
```

The `src/barnaby/sounds.py` mappings are `middle_finger -> growl.wav` and `peace -> sexy.wav`. Add or change mappings in the `EFFECTS` dictionary there. Background music playback is **not yet implemented**.

Speech and effects share one audio worker, so they never play over each other. A held gesture triggers once and must disappear for 0.5 seconds before it can trigger again. Facial-expression changes do not retrigger the held hand gesture. A gesture attempted during its cooldown is ignored until it is released and performed again.

The queue holds at most two waiting effects and the newest waiting observation. Duplicates are ignored. A fresh waiting observation always plays next, ahead of all waiting effects; replacing its text preserves that priority. Expired items are discarded, and audio already playing finishes without interruption. The silence gap still applies between playback items.

| Option | Default | Meaning |
| --- | --- | --- |
| `--observation-expiry` | `8` | Maximum seconds from camera capture to speech starting, including Gemini latency and Piper/SoX preparation |
| `--effect-expiry` | `5` | Maximum seconds an effect can wait before starting |
| `--effect-cooldown` | `5` | Seconds after an effect finishes before the same effect can trigger again |
| `--audio-gap` | `1` | Minimum seconds of silence between playback items |
| `--gesture-release` | `0.5` | Seconds a gesture/expression must be absent to rearm its trigger |

For more frequent observations, add `--scene-interval 5 --observation-expiry 8` to your launch command. The request interval defaults to 10 seconds; it is separate from expiry. Only one Gemini request runs at once, and new requests wait while an observation is queued, being prepared, or being spoken. Sound effects alone do not prevent requests. If Gemini plus speech preparation takes longer than the expiry, that observation is skipped; choose an expiry between 5 and 10 seconds to match the Pi's measured response times.

On the Pi, use PipeWire to route **Piper's `pw-play` audio** to the right output:

```bash
wpctl status
```

Under **Audio → Sinks**, find your Bluetooth speaker or desired HDMI/USB output. Then:

```bash
wpctl set-default SINK_ID
pw-play sounds/effects/growl.wav
```

Replace `SINK_ID` with the number shown by `wpctl status`. If the speaker is not listed, first connect it in Raspberry Pi OS Bluetooth settings. `pygame` uses SDL's audio backend, which may need the same selected system output; if it chooses the wrong backend, try running Barnaby as `SDL_AUDIODRIVER=pulse python -m barnaby ...` from a logged-in desktop/audio session.

## 6. Enroll known faces (with permission)

Face recognition uses **YuNet + SFace** locally. The database is `data/faces.db` and stores numerical face embeddings. A face must be sufficiently clear, at least ~75 pixels across its smaller bounding-box dimension, and be **the only detected face** in the enrollment image. Start from the project root.

### Enroll from the webcam

```bash
python -m barnaby.face_memory enroll --name 'Alex' --source 0 --samples 5
```

Press **Space** five times, changing head angle and lighting between captures. **Q** cancels the enrollment.

### Enroll from a folder of photographs

```bash
mkdir -p pictures/Alex
# Add Alex's JPG/PNG photos under pictures/Alex/
python -m barnaby.face_memory enroll \
  --name 'Alex' \
  --images pictures/Alex/*.jpg pictures/Alex/*.png
```

You can also use uppercase filename globs if needed (`*.JPG`, `*.PNG`). **Paths are relative to the current directory**, so either run from the repo root as shown or adjust paths accordingly. Re-enrolling the same name **replaces** its prior samples.

For HEIC/HEIF photographs copied from an iPad (Linux):

```bash
sudo apt install -y libheif-examples
for file in pictures/Alex/*.HEIC; do
  [ -f "$file" ] || continue
  heif-convert "$file" "${file%.*}.jpg"
done
```

Run the conversion from the project root. If already inside `pictures/Alex/`, change the glob to `*.HEIC`. JPEG or PNG is needed for the existing `cv2.imread` enrollment flow.

### List, test, and remove people

```bash
python -m barnaby.face_memory list
python -m barnaby.face_memory test --source 0
python -m barnaby.face_memory delete --name 'Alex'
```

`test` displays the recognized name and cosine similarity. The default acceptance threshold is `0.50`, and `--threshold` lets you adjust it for testing. **Face matching is probabilistic**; test on different people and lighting conditions, and treat uncertain matches as unknown. During the main Barnaby run, a person must be matched in two consecutive checks before the name is accepted. The current implementation handles **one face at a time**.

## 7. Start the complete Barnaby pipeline

With a USB webcam, enrolled people, Gemini configured, and working audio:

```bash
python -m barnaby \
  --source 0 \
  --faces \
  --face-interval 1.5 \
  --face-threshold 0.50 \
  --scene \
  --scene-model gemini-3.5-flash-lite \
  --scene-interval 5 \
  --observation-expiry 8 \
  --scene-timeout 90 \
  --scene-image-width 320 \
  --threads 2
```

- `--faces` loads enrolled identities from `data/faces.db` and runs SFace at the face interval.
- When Barnaby has a confirmed name, the Gemini request **includes that name** in the text prompt alongside the camera image. If the identity is unknown, it does not supply a name.
- `--scene` submits frames to Gemini. Requests are spaced by at least `--scene-interval` and wait for any pending/active spoken observation. API latency affects actual response timing; observations expire from camera capture time.
- `--threads` controls OpenCV's CPU thread setting; it does not allocate Gemini or Piper worker threads. Start with 2 on a Pi 5 and measure performance before increasing it.
- Piper speaks the returned dialogue if `BARNABY_TTS=1` is set.
- Stable local gestures and expressions can trigger prerecorded sounds through pygame.

Run without cloud access:

```bash
python -m barnaby --source 0 --faces --threads 2
```

Run without a visible preview (e.g., over SSH):

```bash
python -m barnaby --source 0 --faces --scene \
  --scene-model gemini-3.5-flash-lite --headless
```

**Audio caveat:** PipeWire and SDL generally depend on an active user audio session. A headless SSH run may require audio-session configuration; don't run Barnaby as root just to fix audio routing.

## 8. Files you will edit

| File | Purpose |
| --- | --- |
| `src/barnaby/app.py` | Camera loop, CLI arguments, model coordination |
| `src/barnaby/gestures.py` | Hand-landmark gesture rules (`shaka`, `horns`, etc.) |
| `src/barnaby/vision.py` | Hand, face and expression inference |
| `src/barnaby/face_memory.py` | Face enrollment, SFace matching and SQLite |
| `src/barnaby/scene.py` | Gemini scene prompt, recognized-name context |
| `src/barnaby/speech.py` | Piper TTS, SoX effects and `pw-play` |
| `src/barnaby/sounds.py` | Prerecorded sound mappings and blocking playback |
| `src/barnaby/audio_coordinator.py` | Shared audio queue, expiry, cooldowns, release gates and playback worker |
| `src/barnaby/reactions.py` | Gesture/expression events, speech and sounds |
| `src/barnaby/models.py` | Download/verify the four pinned baseline models |
| `.env` | Local Gemini API key and TTS setting; never commit |

### Updating an existing installation

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip install piper-tts google-genai python-dotenv pygame
```

If a new version adds another model or dependency, follow its release instructions. Your `.env`, enrollment photos and database should be treated as machine-local private data rather than being overwritten by Git.

### Test audio scheduling without installing dependencies

These tests use a simulated clock, fake playback and mocked hardware libraries. Python's `-S` option disables site packages, verifying that the audio logic needs no project dependencies.

```powershell
# Windows PowerShell, from the repository root
$env:PYTHONPATH = "src"
python -S -m unittest discover -s tests -p test_audio.py -v
```

```bash
# Linux / Raspberry Pi
PYTHONPATH=src python3 -S -m unittest discover -s tests -p test_audio.py -v
```

Actual camera, Piper performance and speaker playback still need verification on the Pi.

## 9. Important privacy/security cleanup

**The current repository snapshot contains tracked `data/faces.db` and photos in `pictures/`.** Adding paths to `.gitignore` does **not** remove already tracked files. Face embeddings are biometric data, and original photos should not be publicly distributed without the person's permission. From a trusted development machine, after making a secure backup if needed:

```bash
printf '\ndata/\npictures/\n' >> .gitignore
git rm -r --cached --ignore-unmatch data pictures
git add .gitignore
git commit -m 'Stop tracking face data and enrollment photos'
git push
```

`--cached` removes the files from future Git snapshots **without deleting the local copies**. **It does not erase them from earlier public commits/history**; if they were published, address the history separately and treat the data as already exposed. Future fresh clones will need people to enroll again or have `faces.db` securely transferred. Never commit `.env`, API keys, photos, or face embeddings. Enroll only people who have agreed to be recognized, and provide a way to remove their records.

Because this Pi previously had unexpected SSH activity, secure/rebuild it before storing more face data or reusing API keys. Avoid publicly exposing password-authenticated root SSH.

## 10. Troubleshooting

| Problem | Try |
| --- | --- |
| `No module named dotenv / pygame / piper / google.genai` | Activate `.venv`; install the extra Python packages in step 1 |
| `pygame.error: Couldn't open audio device` | Select an audio output, start from an active desktop audio session, or temporarily use `SDL_AUDIODRIVER=dummy` for silent model checks |
| `No such file or directory: pwplay` | The executable is **`pw-play`**; install `pipewire-bin` and check `speech.py` |
| Sound plays through the monitor | `wpctl status`, then `wpctl set-default SINK_ID` for Bluetooth/USB |
| `GEMINI_API_KEY must be set` | Put it in `.env` at the repo root; check that `python-dotenv` is installed |
| `Missing model: ...SFace...` | Download SFace in step 2; it is Git-ignored |
| `Could not open camera/video` | `v4l2-ctl --list-devices`; change `--source` |
| Images skipped during enrollment | Use JPG/PNG, good lighting, one face in frame, and correct relative paths |
| Face not recognized | Try several higher-quality enrollment images and inspect similarity with `face_memory test` |
| CSI camera fails due to OpenCV version | Check `python -c 'import cv2; print(cv2.__version__)'`; code requires 4.10+ |
| Voice model found but no audio | Check `sox`, `pw-play`, `wpctl status`, and `BARNABY_TTS=1` |

To inspect every supported launch flag:

```bash
python -m barnaby --help
python -m barnaby.face_memory --help
```

---

**Status:** Setup documented from the currently committed repository code. Hardware-specific camera, audio and library combinations still require on-device verification. The current project does not yet include a relay/GPIO-control implementation or automatic music-looping logic in the committed sound module.
