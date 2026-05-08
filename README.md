# Billiards World-Model Training Environment

A 2D billiards simulator built with **Pygame** that acts as a training
environment for a world model (LeWM). A scripted geometric agent plays
the game automatically and all episodes are saved to a compact HDF5 dataset.

---

## Project structure

```
billiards-worldmodel/
├── game.py           # Gymnasium-compatible Pygame environment
├── agent.py          # Scripted geometric agent (no ML)
├── collect_data.py   # Runs agent, records data → HDF5
├── play.py           # Interactive human play (no recording)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Installation

```bash
# (Recommended) create a virtual environment first
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

---

## Running each file

### 1. `play.py` — Human interactive mode
```bash
python play.py
```
- **Left-click** anywhere on the table to shoot the cue ball toward the cursor.
- Press **R** to reset the episode.
- Press **ESC** or **Q** to quit.
- No data is saved.

---

### 2. `collect_data.py` — Collect expert dataset
```bash
# Default: 5000 episodes → billiards_expert_train.h5
python collect_data.py

# Custom number of episodes and output file
python collect_data.py --episodes 100 --out test_run.h5

# Fix random seed for reproducibility
python collect_data.py --seed 0
```

Progress is printed every 500 episodes. The script prints a final summary:
```
✓ Saved 5000 episodes (348212 frames) to 'billiards_expert_train.h5'
  Success rate : 87.3%
  Elapsed time : 142.0s  (35.2 ep/s)
```

---

### 3. `game.py` — Gymnasium environment (import or quick smoke-test)
```bash
python game.py        # runs a 10-step smoke test with random actions
```
Or use it as a library:
```python
from game import BilliardsEnv

env = BilliardsEnv(render_mode="rgb_array")
obs, info = env.reset(seed=0)
for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()   # numpy array (512, 512, 3) uint8
    if terminated or truncated:
        break
env.close()
```

---

### 4. `agent.py` — Scripted agent (import or quick test)
```bash
python agent.py       # prints 5 sample actions
```

---

## HDF5 dataset format

| Dataset      | Shape                           | dtype   | Description                        |
|--------------|---------------------------------|---------|------------------------------------|
| `pixels`     | `(total_frames, 512, 512, 3)`   | uint8   | RGB frames (one per timestep)      |
| `action`     | `(total_frames, 2)`             | float32 | `(dx, dy)` impulse applied to cue  |
| `state`      | `(total_frames, 10)`            | float32 | Full state vector (see below)      |
| `ep_len`     | `(num_episodes,)`               | int32   | Number of frames in each episode   |
| `ep_offset`  | `(num_episodes,)`               | int64   | Start frame index of each episode  |

**State vector layout** (indices 0–9):

| Index | Value               |
|-------|---------------------|
| 0–1   | cue ball position   |
| 2–3   | cue ball velocity   |
| 4–5   | target ball position|
| 6–7   | target ball velocity|
| 8–9   | nearest pocket (x,y)|

### Reading the dataset
```python
import h5py, numpy as np

with h5py.File("billiards_expert_train.h5", "r") as f:
    pixels    = f["pixels"]     # lazy; index as needed
    actions   = f["action"][:]
    states    = f["state"][:]
    ep_len    = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]

# Reconstruct episode 42
i   = ep_offset[42]
n   = ep_len[42]
frames = pixels[i : i + n]      # shape (n, 512, 512, 3)
```

---

## Environment details

| Parameter        | Value                  |
|------------------|------------------------|
| Window size      | 512 × 512 px           |
| Ball radius      | 15 px                  |
| Pocket radius    | 20 px                  |
| Friction         | 0.985 per step         |
| Max steps/ep     | 300                    |
| Success condition| Target ball enters pocket |

---

## Agent strategy

The scripted agent uses the **ghost-ball** technique from billiards:

1. Identify the pocket nearest to the target ball.
2. Compute the *ghost-ball* position G — where the cue ball's centre
   must be at impact for the target to travel toward that pocket:

$$G = T - \hat{u}_{PT} \cdot 2R$$

where $T$ is the target position, $\hat{u}_{PT}$ is the unit vector
from pocket to target, and $R$ is the ball radius.

3. Apply an impulse in the direction $G - C_\text{cue}$, scaled by
   `impulse_strength`, with small Gaussian noise for episode variety.
4. The impulse is only applied when the cue ball is nearly stationary
   (speed < 0.5 px/step) to avoid stacking impulses mid-roll.
