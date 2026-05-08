"""
collect_data.py - Data Collection Script for Billiards World Model

Runs the scripted agent in the billiards environment for N episodes,
records pixels / actions / states at every timestep, then saves
everything to an HDF5 file in the flat LeWM format.

HDF5 layout
-----------
    pixels    : (total_frames, 96, 96, 3)  uint8   ← downscaled from 512 for storage
    action    : (total_frames, 2)          float32
    state     : (total_frames, 10)         float32
    ep_len    : (num_episodes,)            int32
    ep_offset : (num_episodes,)            int64

Usage
-----
    python collect_data.py                    # 2000 episodes (default)
    python collect_data.py --episodes 100     # quick test run
    python collect_data.py --out mydata.h5    # custom filename
"""

import argparse
import time

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

from agent import ScriptedAgent
from game import BilliardsEnv, WINDOW_SIZE, RENDER_SIZE, MAX_STEPS

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_EPISODES = 2000
DEFAULT_OUTFILE = "billiards_expert_train.h5"
FRAME_SHAPE = (RENDER_SIZE, RENDER_SIZE, 3)   # 96×96×3
OBS_DIM = 10
ACTION_DIM = 2


def collect(num_episodes: int, outfile: str, seed: int = 42):
    """
    Run the scripted agent for *num_episodes* episodes and save to HDF5.

    A pre-allocated HDF5 dataset is grown incrementally so that memory
    usage stays bounded even for large episode counts.
    """
    rng = np.random.default_rng(seed)
    env = BilliardsEnv(render_mode="rgb_array")
    agent = ScriptedAgent(rng=rng)

    # ── Open HDF5 file with resizable datasets ────────────────────────────────
    with h5py.File(outfile, "w") as f:

        # Chunked, resizable datasets (chunk along time axis)
        chunk_t = min(256, MAX_STEPS)
        ds_pixels = f.create_dataset(
            "pixels",
            shape=(0, *FRAME_SHAPE),
            maxshape=(None, *FRAME_SHAPE),
            dtype="uint8",
            chunks=(chunk_t, *FRAME_SHAPE),
            **hdf5plugin.LZ4(),   # fast LZ4 compression — pixels only
        )
        ds_action = f.create_dataset(
            "action",
            shape=(0, ACTION_DIM),
            maxshape=(None, ACTION_DIM),
            dtype="float32",
            chunks=(chunk_t, ACTION_DIM),
        )
        ds_state = f.create_dataset(
            "state",
            shape=(0, OBS_DIM),
            maxshape=(None, OBS_DIM),
            dtype="float32",
            chunks=(chunk_t, OBS_DIM),
        )

        ep_len_list = []   # length of each episode
        ep_offset_list = []   # starting frame index of each episode
        total_frames = 0
        success_count = 0

        # Pre-allocate per-episode buffers (MAX_STEPS bound; reused each episode)
        buf_pixels = np.empty((MAX_STEPS, *FRAME_SHAPE), dtype=np.uint8)
        buf_actions = np.empty((MAX_STEPS, ACTION_DIM),   dtype=np.float32)
        buf_states = np.empty((MAX_STEPS, OBS_DIM),      dtype=np.float32)

        start_time = time.time()

        pbar = tqdm(range(num_episodes), desc="Collecting",
                    unit="ep", dynamic_ncols=True)
        for ep_idx in pbar:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))

            t = 0
            terminated = truncated = False

            while not (terminated or truncated):
                buf_pixels[t] = env.render()
                buf_actions[t] = agent.act(obs)
                buf_states[t] = obs
                obs, _reward, terminated, truncated, _info = env.step(
                    buf_actions[t])
                t += 1

            # ── Append episode data to HDF5 ───────────────────────────────────
            ep_len = t
            ep_offset_list.append(total_frames)
            ep_len_list.append(ep_len)
            total_frames += ep_len

            ds_pixels.resize(total_frames, axis=0)
            ds_action.resize(total_frames, axis=0)
            ds_state.resize(total_frames,  axis=0)

            t_start = total_frames - ep_len
            ds_pixels[t_start:total_frames] = buf_pixels[:ep_len]
            ds_action[t_start:total_frames] = buf_actions[:ep_len]
            ds_state[t_start:total_frames] = buf_states[:ep_len]

            if terminated:
                success_count += 1

            # ── Live progress bar postfix (updated every episode) ─────────────
            elapsed = time.time() - start_time
            eps_done = ep_idx + 1
            eps_per_s = eps_done / elapsed if elapsed > 0 else 0.0
            pct_ok = 100.0 * success_count / eps_done
            remaining = (num_episodes - eps_done) / \
                eps_per_s if eps_per_s > 0 else 0
            pbar.set_postfix(
                success=f"{success_count}/{eps_done}",
                rate=f"{pct_ok:.0f}%",
                speed=f"{eps_per_s:.1f}ep/s",
                eta=f"{remaining:.0f}s",
            )

        # ── Write index arrays ────────────────────────────────────────────────
        f.create_dataset("ep_len",    data=np.array(
            ep_len_list,    dtype=np.int32))
        f.create_dataset("ep_offset", data=np.array(
            ep_offset_list, dtype=np.int64))

        # ── Summary attributes ────────────────────────────────────────────────
        f.attrs["num_episodes"] = num_episodes
        f.attrs["total_frames"] = total_frames
        f.attrs["success_rate"] = success_count / num_episodes
        f.attrs["window_size"] = WINDOW_SIZE
        f.attrs["max_steps"] = MAX_STEPS

    env.close()
    elapsed = time.time() - start_time
    print(
        f"\n✓ Saved {num_episodes} episodes ({total_frames} frames) to '{outfile}'")
    print(f"  Success rate : {100.0 * success_count / num_episodes:.1f}%")
    print(
        f"  Elapsed time : {elapsed:.1f}s  ({num_episodes / elapsed:.1f} ep/s)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect billiards expert data")
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help=f"Number of episodes to collect (default: {DEFAULT_EPISODES})"
    )
    parser.add_argument(
        "--out", type=str, default=DEFAULT_OUTFILE,
        help=f"Output HDF5 filename (default: {DEFAULT_OUTFILE})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    collect(num_episodes=args.episodes, outfile=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
