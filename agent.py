"""
agent.py - Scripted Geometric Agent for Billiards

The agent computes the ideal cue-ball impulse that will:
  1. Hit the target ball along a line toward the nearest pocket.
  2. Add small random noise so that episodes are varied.

No learning is involved — pure geometry.
"""

import numpy as np
from game import POCKET_POSITIONS, BALL_RADIUS


class ScriptedAgent:
    """
    Geometric billiards agent.

    Strategy
    --------
    Given target position T and nearest pocket P, we compute the
    "ghost ball" position G: the point where the cue ball must be
    located at impact so that the target ball travels toward P.

        G = T  -  (P - T) / |P - T|  *  2 * R

    The agent then applies an impulse in the direction (G - cue_pos),
    scaled by a tunable strength parameter, with Gaussian noise.
    """

    def __init__(
        self,
        impulse_strength: float = 14.0,
        noise_std: float = 0.06,
        rng: np.random.Generator | None = None,
    ):
        """
        Parameters
        ----------
        impulse_strength : float
            Magnitude of the impulse applied per step.
        noise_std : float
            Standard deviation of Gaussian noise added to the direction.
        rng : numpy Generator, optional
            Random number generator for reproducibility.
        """
        self.impulse_strength = impulse_strength
        self.noise_std = noise_std
        self.rng = rng or np.random.default_rng()

    # ── Public API ────────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute the (dx, dy) impulse to apply to the cue ball.

        Parameters
        ----------
        obs : ndarray, shape (10,)
            [cue_x, cue_y, cue_vx, cue_vy,
             target_x, target_y, target_vx, target_vy,
             nearest_pocket_x, nearest_pocket_y]

        Returns
        -------
        action : ndarray, shape (2,) float32
        """
        cue_pos = obs[0:2]
        cue_vel = obs[2:4]
        target_pos = obs[4:6]
        target_vel = obs[6:8]
        pocket = obs[8:10]

        # ── Only shoot when the CUE ball is stationary ────────────────────────
        # Allowing shots while target is still rolling gives more attempts per
        # episode (critical with only 300 steps). The ghost-ball geometry is
        # re-computed each time using the target's current position.
        if np.linalg.norm(cue_vel) > 0.5:
            return np.zeros(2, dtype=np.float32)

        # ── Ghost-ball position ───────────────────────────────────────────────
        ghost = self._ghost_ball_pos(target_pos, pocket)

        # ── Direction from cue ball to ghost ball ─────────────────────────────
        direction = ghost - cue_pos
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            direction = target_pos - cue_pos
            dist = max(np.linalg.norm(direction), 1e-6)

        direction /= dist  # normalise

        # ── Scale impulse by distance so ball reaches ghost position ──────────
        # Minimum strength guarantees contact; scaling by dist adds power for
        # far shots so the target ball reaches the pocket.
        MIN_STRENGTH = 6.0
        scaled_strength = max(
            MIN_STRENGTH, self.impulse_strength * dist / 200.0)

        # ── Add small noise for episode variety ───────────────────────────────
        noise = self.rng.normal(0.0, self.noise_std, size=2)
        direction = direction + noise
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm

        action = (direction * scaled_strength).astype(np.float32)
        return action

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _ghost_ball_pos(target_pos: np.ndarray, pocket: np.ndarray) -> np.ndarray:
        """
        Compute the ghost-ball position: where the cue ball's centre
        must be at the moment of impact so the target travels toward pocket.

            G = T - unit(P - T) * 2R
        """
        pocket_dir = pocket - target_pos
        dist = np.linalg.norm(pocket_dir)
        if dist < 1e-6:
            return target_pos.copy()
        unit = pocket_dir / dist
        return target_pos - unit * (2 * BALL_RADIUS)

    @staticmethod
    def nearest_pocket(target_pos: np.ndarray) -> np.ndarray:
        """Return the pocket position closest to target_pos."""
        dists = np.linalg.norm(POCKET_POSITIONS - target_pos, axis=1)
        return POCKET_POSITIONS[np.argmin(dists)]


# ── Standalone quick-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    from game import BilliardsEnv
    print("Printing 5 sample agent actions …")
    env = BilliardsEnv(render_mode="rgb_array")
    agent = ScriptedAgent(rng=np.random.default_rng(0))
    obs, _ = env.reset(seed=0)
    for i in range(5):
        action = agent.act(obs)
        print(f"  step {i+1}: action={action}")
        obs, *_ = env.step(action)
    env.close()
    print("Agent test passed ✓")
