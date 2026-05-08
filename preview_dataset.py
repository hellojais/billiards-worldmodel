"""
preview_dataset.py - Visualise the billiards HDF5 dataset

Shows:
  1. First frame of 6 randomly chosen episodes in a matplotlib grid.
  2. One full episode played as a matplotlib FuncAnimation.
  3. (--compare) Three episodes animated side by side for comparison.

Usage
-----
    python preview_dataset.py                              # uses default file
    python preview_dataset.py --file test_50f.h5          # custom file
    python preview_dataset.py --episode 3                 # animate episode 3
    python preview_dataset.py --compare 45 1 2            # side-by-side compare
"""

import argparse
import sys

import h5py
import hdf5plugin          # must import to register LZ4 decompressor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


DEFAULT_FILE = "billiards_expert_train.h5"


def load_meta(path: str):
    """Return ep_offset, ep_len arrays and a lazy handle to the file."""
    f = h5py.File(path, "r")
    ep_offset = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    return f, ep_offset, ep_len


# ── Part 1: 6-episode first-frame grid ───────────────────────────────────────

def show_grid(f, ep_offset, ep_len, n: int = 6):
    """Display the first frame of n random episodes in a grid."""
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ep_offset), size=min(
        n, len(ep_offset)), replace=False)
    indices = sorted(indices)

    cols = 3
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for ax_i, ep_i in enumerate(indices):
        frame_idx = int(ep_offset[ep_i])
        frame = f["pixels"][frame_idx]          # (96, 96, 3) uint8
        state = f["state"][frame_idx]
        action = f["action"][frame_idx]

        axes[ax_i].imshow(frame)
        axes[ax_i].set_title(
            f"ep {ep_i}  |  len={ep_len[ep_i]}\n"
            f"cue=({state[0]:.0f},{state[1]:.0f})  "
            f"tgt=({state[4]:.0f},{state[5]:.0f})",
            fontsize=7,
        )
        axes[ax_i].axis("off")

    # Hide any unused axes
    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle("First frame of 6 random episodes",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)


# ── Part 2: Full-episode animation ───────────────────────────────────────────

def animate_episode(f, ep_offset, ep_len, ep_idx: int = 0):
    """Play one episode as a matplotlib FuncAnimation."""
    start = int(ep_offset[ep_idx])
    n = int(ep_len[ep_idx])
    frames = f["pixels"][start: start + n]   # (n, 96, 96, 3)
    actions = f["action"][start: start + n]
    states = f["state"][start: start + n]

    fig, ax = plt.subplots(figsize=(4, 4.5))
    ax.axis("off")
    im = ax.imshow(frames[0])
    title = ax.set_title("")

    success = ep_len[ep_idx] < 300

    def update(t):
        im.set_data(frames[t])
        title.set_text(
            f"Episode {ep_idx}  |  step {t+1}/{n}  "
            f"({'SUCCESS' if success and t == n-1 else 'running' if t < n-1 else 'timeout'})\n"
            f"action=({actions[t, 0]:+.2f}, {actions[t, 1]:+.2f})  "
            f"cue=({states[t, 0]:.0f},{states[t, 1]:.0f})"
        )
        return [im, title]

    ani = animation.FuncAnimation(
        fig, update,
        frames=n,
        interval=40,      # ~25 fps
        blit=False,
        repeat=True,
    )

    fig.suptitle(
        f"Full episode animation  |  {'✓ POTTED' if success else '✗ timeout'}",
        fontsize=10, fontweight="bold",
        color="green" if success else "red",
    )
    plt.tight_layout()
    return ani   # keep reference alive


# ── Part 3: Side-by-side comparison of multiple episodes ─────────────────────

def compare_episodes(f, ep_offset, ep_len, ep_indices: list[int]):
    """Animate multiple episodes side by side in a single figure."""
    n_eps = len(ep_indices)

    # Load all frame buffers upfront
    all_frames = []
    all_actions = []
    all_states = []
    max_len = 0

    for ep_i in ep_indices:
        start = int(ep_offset[ep_i])
        n = int(ep_len[ep_i])
        all_frames.append(f["pixels"][start: start + n])
        all_actions.append(f["action"][start: start + n])
        all_states.append(f["state"][start: start + n])
        max_len = max(max_len, n)

    # Labels for each panel
    def panel_label(ep_i):
        l = int(ep_len[ep_i])
        outcome = "✓ SUCCESS" if l < 300 else "✗ TIMEOUT"
        return f"Ep {ep_i}  |  {l} steps  |  {outcome}"

    fig, axes = plt.subplots(1, n_eps, figsize=(n_eps * 4, 4.6))
    if n_eps == 1:
        axes = [axes]

    # Panel border colours: green for success, red for timeout
    border_colors = ["green" if ep_len[i] < 300 else "red" for i in ep_indices]
    for ax, col in zip(axes, border_colors):
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(3)

    ims = [ax.imshow(all_frames[k][0]) for k, ax in enumerate(axes)]
    titles = [ax.set_title(panel_label(ep_i), fontsize=8, color=border_colors[k])
              for k, (ax, ep_i) in enumerate(zip(axes, ep_indices))]
    steps = [ax.text(2, 6, "step 1", color="white", fontsize=7,
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
             for ax in axes]

    for ax in axes:
        ax.axis("off")

    fig.suptitle("Episode Comparison  |  green border = success  |  red = timeout",
                 fontsize=10, fontweight="bold")

    def update(t):
        artists = []
        for k, ep_i in enumerate(ep_indices):
            ep_t = min(t, len(all_frames[k]) - 1)   # hold last frame when done
            ims[k].set_data(all_frames[k][ep_t])
            done = ep_t >= len(all_frames[k]) - 1
            outcome = ("✓ POTTED" if ep_len[ep_i] <
                       300 else "✗ TIMEOUT") if done else "…"
            steps[k].set_text(f"step {ep_t+1}/{ep_len[ep_i]}  {outcome}")
            artists += [ims[k], steps[k]]
        return artists

    ani = animation.FuncAnimation(
        fig, update,
        frames=max_len,
        interval=40,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    return ani


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preview billiards HDF5 dataset")
    parser.add_argument("--file",    default=DEFAULT_FILE,
                        help="HDF5 file path")
    parser.add_argument("--episode", type=int, default=None,
                        help="Episode index to animate (default: first success)")
    parser.add_argument("--compare", type=int, nargs="+", default=None,
                        help="Animate N episodes side by side, e.g. --compare 45 1 2")
    args = parser.parse_args()

    print(f"Loading {args.file} …")
    try:
        f, ep_offset, ep_len = load_meta(args.file)
    except FileNotFoundError:
        print(f"  ERROR: '{args.file}' not found.")
        print(
            "  Run:  python collect_data.py --episodes 50 --out billiards_expert_train.h5")
        sys.exit(1)

    n_eps = len(ep_offset)
    success_mask = ep_len < 300
    print(f"  Episodes : {n_eps}")
    print(
        f"  Success  : {success_mask.sum()} ({100*success_mask.mean():.0f}%)")
    print(f"  Frames   : {ep_len.sum()}")
    print(f"  Frame shape : {f['pixels'][0].shape}")

    # ── Part 3: side-by-side comparison (takes priority if --compare given) ──
    if args.compare:
        print(f"\nComparing episodes {args.compare} side by side …")
        ani = compare_episodes(f, ep_offset, ep_len, args.compare)
        plt.show()
        f.close()
        return

    # ── Part 1: grid ─────────────────────────────────────────────────────────
    show_grid(f, ep_offset, ep_len, n=6)

    # ── Part 2: animation ────────────────────────────────────────────────────
    if args.episode is not None:
        ep_idx = args.episode
    else:
        # Pick the first successful episode for a satisfying animation
        successes = np.where(success_mask)[0]
        ep_idx = int(successes[0]) if len(successes) > 0 else 0

    print(f"\nAnimating episode {ep_idx}  (len={ep_len[ep_idx]}, "
          f"{'success' if success_mask[ep_idx] else 'timeout'}) …")
    ani = animate_episode(f, ep_offset, ep_len, ep_idx)

    plt.show()   # blocks until both windows are closed
    f.close()


if __name__ == "__main__":
    main()
