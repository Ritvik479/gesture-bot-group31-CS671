## File Structure
gesture-bot/
│
├── data/
│   ├── raw/                        # .npy files straight from collection script
│   └── processed/                  # normalized, windowed, train/val split
│
├── collection/
│   ├── collect_gestures.py         # webcam + MediaPipe data collection tool
│   └── verify_dataset.py           # visualizes saved sequences to sanity check
│
├── training/
│   ├── preprocess.py               # normalization, sliding window, train/val split
│   ├── model.py                    # LSTM architecture definition
│   ├── train.py                    # training loop, saves best model
│   └── evaluate.py                 # confusion matrix, per-class accuracy
│
├── inference/
│   ├── gesture_recognizer.py       # loads model, runs sliding window on live feed
│   └── command_mapper.py           # gesture label → robot action string
│
├── simulation/
│   ├── humanoid.xml                # MuJoCo humanoid model (or fetch built-in)
│   └── sim_controller.py           # receives action strings, applies to MuJoCo
│
├── display/
│   └── compositor.py              # tiles webcam + skeleton + sim into one window
│
├── main.py                         # entry point, wires everything together
├── config.py                       # all constants in one place
├── requirements.txt
└── README.md

## Explanation of file contents
### `config.py`
Single source of truth for every constant. Gesture class names and their integer labels, number of frames per window (30), confidence threshold for command firing (e.g. 0.85), webcam resolution, MuJoCo model path, action mappings. Every other file imports from here — nothing is hardcoded anywhere else.
### `collection/collect_gestures.py`
Opens webcam, runs MediaPipe Pose, overlays skeleton. On keypress (1–6), starts a 3-second countdown, then records exactly 30 frames of 99-dim landmark vectors, saves as a `.npy` file to `data/raw/`, displays a green "SAVED ✓" confirmation. Filenames auto-labeled with class name, person ID, and timestamp.
### `collection/verify_dataset.py`
Loads a saved `.npy` sequence and replays it frame-by-frame on a blank canvas by redrawing the skeleton from the landmark vectors. Lets you visually confirm that a saved sample actually looks like the intended gesture.
### `training/preprocess.py`
Loads all `.npy` files from `data/raw/`, applies landmark normalization (hip-centered, scale-normalized), assembles into `[N, 30, 99]` arrays, creates an idle/null class from random non-gesture frames, does an 80/20 train/val split stratified by person ID, saves to `data/processed/`.
### `training/model.py`
Defines the LSTM in Keras. Two LSTM layers (64 units each), dropout between them, dense softmax output with N+1 classes (5 gestures + null). Kept intentionally small — this model trains in minutes.
### `training/train.py`
Loads processed data, compiles and trains the model, uses early stopping and model checkpointing to save the best weights to `models/best_model.keras`. Plots training/val accuracy curves.
### `training/evaluate.py`
Loads the saved model and val set, prints a confusion matrix and per-class accuracy. Tells you which gestures are being confused with each other so you can go collect better data for those classes.
### `inference/gesture_recognizer.py`
Maintains a rolling 30-frame buffer of incoming MediaPipe landmarks. On every frame, runs the LSTM on the current window. Returns gesture label + confidence. Only emits a gesture event if confidence exceeds the threshold in `config.py` and the gesture has been held for at least 0.5 seconds (debounce).
### `inference/command_mapper.py`
A simple dictionary lookup: gesture label → action string (e.g. `"RAISE_ARM" → "MOVE_FORWARD"`). Includes the confidence gate logic. Stateless — just maps and returns.
### `simulation/sim_controller.py`
Initializes MuJoCo, loads the humanoid XML, exposes a `send_action(action_string)` function. Each action string maps to a pre-baked motion primitive (a short sequence of joint position targets). Runs the sim step loop in a background thread so it doesn't block the webcam pipeline.
### `display/compositor.py`
Takes three inputs each frame: the raw webcam frame with skeleton drawn, the current gesture label + confidence bar, and the MuJoCo rendered frame. Tiles them side by side into a single OpenCV window. Also shows a small command log at the bottom showing the last 3 actions fired.
### `main.py`
The only file you run. Initializes all modules, starts the sim controller thread, opens the webcam loop, and on each frame: passes it through MediaPipe → gesture recognizer → command mapper → sim controller → compositor → display.

## Plan of Action
**Step 1 — Set up repo and config (1 person, ~1 hour)**
Create the folder structure, initialize git, write `requirements.txt`, and populate `config.py` with all gesture names, constants and mappings. Push to GitHub. Everyone clones. This unblocks everything else.

**Step 2 — Write `collect_gestures.py` (1 person, ~2 hours)**
This is the first priority. Nothing else can start without data. Get the webcam + MediaPipe skeleton overlay running, add the keypress trigger, countdown, 30-frame capture, and save logic. Test it personally, confirm `.npy` files are saving correctly.

**Step 3 — Write `verify_dataset.py` (same person, ~1 hour)**
Immediately after the collector works, write the verifier. You want to confirm that what you saved actually looks right before recording 200 samples of garbage.

**Step 4 — Record the dataset (whole team, ~2–3 hours)**
All 9 members take turns in front of the webcam. Each person records ~30–40 samples of each gesture class including the idle/null class. Vary lighting, distance, and clothing. One person oversees and spot-checks using the verifier script throughout. Target: 250–300 samples per class.

**Step 5 — Write `preprocess.py` and `model.py` (1–2 people, parallel with Step 4)**
While data is being recorded, another subteam writes the preprocessing pipeline and LSTM architecture. They can test it on whatever partial data is available.

**Step 6 — Write `train.py` and `evaluate.py`, then train (1–2 people, ~2 hours)**
Once the dataset is complete and preprocessing is verified, run the first training. Check the confusion matrix. If any class is underperforming, go back and collect 50 more samples of that class specifically and retrain.

**Step 7 — Write `gesture_recognizer.py` (1 person, ~2 hours)**
Build the rolling buffer and debounce logic. Test it in isolation — print the gesture label and confidence to console on a live webcam feed. Do not connect it to anything yet. Confirm that gestures are being detected reliably before wiring downstream.

**Step 8 — Write `sim_controller.py` (1–2 people, ~3 hours, parallel with Steps 6–7)**
This is the most independent module and can be developed entirely in isolation. Get MuJoCo running with the humanoid, define the 5 motion primitives, and test them by calling `send_action()` manually from a Python shell. Don't wait for the gesture pipeline to be ready.

**Step 9 — Write `command_mapper.py` (1 person, ~30 minutes)**
Trivial once config is done. Just the dictionary and the confidence gate.

**Step 10 — Write `compositor.py` (1 person, ~1–2 hours)**
Build the tiled display. Can be tested with dummy/placeholder frames before the real pipeline is connected.

**Step 11 — Write `main.py` and integrate (2 people, ~2–3 hours)**
Wire all modules together. This is where bugs surface at the boundaries between modules. Expect to spend time here. The threading between the webcam loop and MuJoCo loop especially needs care.

**Step 12 — End-to-end testing and tuning (~3–4 hours)**
Run the full pipeline in the actual demo environment (lighting, background). Tune the confidence threshold and debounce timing in `config.py`. Test all 5 gestures repeatedly. Fix whatever breaks.

**Step 13 — Record backup demo video (1 hour)**
Non-negotiable. Record a clean run of the full demo. Upload to Google Drive. If anything goes wrong during the live demo, you have this.

**Step 14 — Prep README, architecture diagram, and rehearse demo (1–2 hours)**
Write setup instructions in the README. Assign demo roles: one person performs gestures, one narrates. Do 3 full dry runs.

## Pipeline in plain english
Webcam → MediaPipe extracts body skeleton → LSTM classifies gesture → mapped to robot command → MuJoCo simulation reacts → all shown in one split-screen window

## Ground rules
1. Nothing is hardcoded — everything goes through config.py
2. Each module must work in isolation before integration
3. We record the backup demo video no matter what

Note: Use `pytorch` instead of `tensorflow` as the deep learning library. This is the library written in `requirements.txt` anyway. `mujoco` only works with python versions <`3.12`.
## Code to create virtual environment and install dependencies
```python
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Let's go 🚀
