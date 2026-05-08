"""
game.py - Billiards Pygame Environment (Gymnasium-compatible)

Provides a 2D billiards environment with:
- A white cue ball and a red target ball
- Four corner pockets
- Elastic collision physics with friction
- Gymnasium-style reset() / step() / render() interface
"""

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 512   # internal physics / display resolution
RENDER_SIZE = 96    # output frame size saved to dataset (downscaled from 512)
BALL_RADIUS = 15
POCKET_RADIUS = 20
FRICTION = 0.96           # velocity multiplier per step (< 1 → slow down)
MAX_STEPS = 300
FPS = 60

# Table boundaries (balls bounce off the inner walls)
WALL_LEFT = BALL_RADIUS
WALL_RIGHT = WINDOW_SIZE - BALL_RADIUS
WALL_TOP = BALL_RADIUS
WALL_BOTTOM = WINDOW_SIZE - BALL_RADIUS

# Pocket positions (four corners)
POCKET_POSITIONS = np.array([
    [POCKET_RADIUS,              POCKET_RADIUS],               # top-left
    [WINDOW_SIZE - POCKET_RADIUS, POCKET_RADIUS],              # top-right
    [POCKET_RADIUS,              WINDOW_SIZE - POCKET_RADIUS],  # bottom-left
    [WINDOW_SIZE - POCKET_RADIUS, WINDOW_SIZE - POCKET_RADIUS],  # bottom-right
], dtype=np.float32)

# Colors
COLOR_FELT = (34,  139,  34)   # green felt
COLOR_CUE = (255, 255, 255)   # white
COLOR_TARGET = (220,  30,  30)   # red
COLOR_POCKET = (10,   10,  10)   # near-black
COLOR_OUTLINE = (0,     0,   0)


class BilliardsEnv(gym.Env):
    """
    2D billiards environment.

    Observation space:  10-dimensional state vector
        [cue_x, cue_y, cue_vx, cue_vy,
         target_x, target_y, target_vx, target_vy,
         nearest_pocket_x, nearest_pocket_y]

    Action space: 2-dimensional continuous impulse (dx, dy)
        applied to the cue ball each step.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": FPS}

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0, high=float(WINDOW_SIZE),
            shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(2,), dtype=np.float32
        )

        # Pygame surfaces (lazily initialised)
        self._screen = None
        self._surface = None
        self._clock = None

        # Internal state
        self._step_count = 0
        self.cue_pos = np.zeros(2, dtype=np.float32)
        self.cue_vel = np.zeros(2, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.target_vel = np.zeros(2, dtype=np.float32)

        # Pocket-sink animation state (used by _draw_frame)
        self._target_potted = False
        self._pocket_sink_pos = np.zeros(2, dtype=np.float32)
        self._sink_radius = float(BALL_RADIUS)

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset episode: randomise ball positions, zero velocities."""
        super().reset(seed=seed)
        rng = self.np_random  # seeded RNG provided by Gymnasium

        # Place cue ball in the lower half, avoiding pockets
        self.cue_pos = self._random_position(
            rng, y_range=(WINDOW_SIZE // 2, WINDOW_SIZE - 60))
        self.cue_vel = np.zeros(2, dtype=np.float32)

        # Place target ball in the upper half, far enough from cue ball
        for _ in range(200):
            self.target_pos = self._random_position(
                rng, y_range=(60, WINDOW_SIZE // 2))
            if np.linalg.norm(self.target_pos - self.cue_pos) > 4 * BALL_RADIUS:
                break
        self.target_vel = np.zeros(2, dtype=np.float32)

        self._step_count = 0
        self._target_potted = False
        self._pocket_sink_pos = np.zeros(2, dtype=np.float32)
        self._sink_radius = float(BALL_RADIUS)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        """
        Apply action (impulse) to cue ball, advance physics one step.

        Returns: obs, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.float32)

        # Apply impulse only when both balls are roughly stationary
        # (allows the agent to "shoot" each step; works fine for scripted agent)
        self.cue_vel += action

        # Physics update
        self._update_physics()
        self._step_count += 1

        # Check termination conditions
        terminated = self._check_pocket(
            self.target_pos)  # target potted → success
        truncated = self._step_count >= MAX_STEPS

        # Record sink position for the shrink animation
        if terminated:
            dists = np.linalg.norm(POCKET_POSITIONS - self.target_pos, axis=1)
            self._pocket_sink_pos = POCKET_POSITIONS[np.argmin(dists)].copy()
            self._target_potted = True
            self._sink_radius = float(BALL_RADIUS)

        reward = 1.0 if terminated else 0.0

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the current frame. Returns an RGB numpy array at RENDER_SIZE."""
        self._ensure_pygame()
        self._draw_frame()

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(FPS)

        # Downscale from WINDOW_SIZE to RENDER_SIZE using pygame's fast transform,
        # then return as a numpy array in (H, W, 3) orientation.
        small = pygame.transform.scale(
            self._surface, (RENDER_SIZE, RENDER_SIZE))
        return pygame.surfarray.array3d(small).transpose(1, 0, 2)

    def close(self):
        """Shut down Pygame if it was initialised."""
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._surface = None
            self._clock = None

    # ── Physics helpers ───────────────────────────────────────────────────────

    def _update_physics(self):
        """Move balls, handle wall bounces and ball–ball collision."""
        # Move balls
        self.cue_pos += self.cue_vel
        self.target_pos += self.target_vel

        # Wall bounces for each ball
        self._bounce_wall(self.cue_pos,    self.cue_vel)
        self._bounce_wall(self.target_pos, self.target_vel)

        # Ball–ball elastic collision
        self._resolve_collision()

        # Apply friction
        self.cue_vel *= FRICTION
        self.target_vel *= FRICTION

        # Clamp near-zero velocities to exactly zero (avoids drift)
        STOP_THRESH = 0.01
        if np.linalg.norm(self.cue_vel) < STOP_THRESH:
            self.cue_vel[:] = 0.0
        if np.linalg.norm(self.target_vel) < STOP_THRESH:
            self.target_vel[:] = 0.0

    def _bounce_wall(self, pos: np.ndarray, vel: np.ndarray):
        """Reflect ball velocity off table walls and clamp position."""
        if pos[0] < WALL_LEFT:
            pos[0] = WALL_LEFT
            vel[0] = abs(vel[0])
        elif pos[0] > WALL_RIGHT:
            pos[0] = WALL_RIGHT
            vel[0] = -abs(vel[0])

        if pos[1] < WALL_TOP:
            pos[1] = WALL_TOP
            vel[1] = abs(vel[1])
        elif pos[1] > WALL_BOTTOM:
            pos[1] = WALL_BOTTOM
            vel[1] = -abs(vel[1])

    def _resolve_collision(self):
        """1D elastic collision along the line of centres (equal masses)."""
        delta = self.target_pos - self.cue_pos
        dist = np.linalg.norm(delta)
        min_dist = 2 * BALL_RADIUS

        if dist < min_dist and dist > 1e-6:
            # Separate overlapping balls
            normal = delta / dist
            overlap = min_dist - dist
            self.cue_pos -= normal * overlap * 0.5
            self.target_pos += normal * overlap * 0.5

            # Exchange velocity components along the normal (equal mass)
            v_cue_n = np.dot(self.cue_vel,    normal)
            v_target_n = np.dot(self.target_vel, normal)
            self.cue_vel += (v_target_n - v_cue_n) * normal
            self.target_vel += (v_cue_n - v_target_n) * normal

    def _check_pocket(self, pos: np.ndarray) -> bool:
        """Return True if the ball is close enough to any pocket."""
        for pocket in POCKET_POSITIONS:
            if np.linalg.norm(pos - pocket) < POCKET_RADIUS:
                return True
        return False

    # ── Observation helpers ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return 10-dim state vector."""
        nearest_pocket = self._nearest_pocket(self.target_pos)
        return np.concatenate([
            self.cue_pos,    self.cue_vel,
            self.target_pos, self.target_vel,
            nearest_pocket,
        ]).astype(np.float32)

    def _nearest_pocket(self, pos: np.ndarray) -> np.ndarray:
        """Return the (x, y) of the pocket nearest to *pos*."""
        dists = np.linalg.norm(POCKET_POSITIONS - pos, axis=1)
        return POCKET_POSITIONS[np.argmin(dists)]

    # ── Rendering helpers ─────────────────────────────────────────────────────

    def _ensure_pygame(self):
        """Initialise Pygame lazily."""
        if self._surface is not None:
            return
        pygame.init()
        self._surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        self._clock = pygame.time.Clock()
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Billiards")

    def _draw_frame(self):
        """Draw all game elements onto the offscreen surface."""
        surf = self._surface
        surf.fill(COLOR_FELT)

        # Draw pockets
        for px, py in POCKET_POSITIONS:
            pygame.draw.circle(surf, COLOR_POCKET,
                               (int(px), int(py)), POCKET_RADIUS)

        # Draw target ball (red) — or shrink-into-pocket animation
        if self._target_potted:
            # Animate ball shrinking into the pocket centre
            if self._sink_radius > 0:
                sx = int(self._pocket_sink_pos[0])
                sy = int(self._pocket_sink_pos[1])
                r = max(1, int(self._sink_radius))
                pygame.draw.circle(surf, COLOR_TARGET,  (sx, sy), r)
                pygame.draw.circle(surf, COLOR_OUTLINE, (sx, sy), r, 1)
                self._sink_radius -= 1.5   # shrink 1.5 px per frame
            # once radius ≤ 0 the ball simply isn't drawn → gone
        else:
            tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
            pygame.draw.circle(surf, COLOR_TARGET,  (tx, ty), BALL_RADIUS)
            pygame.draw.circle(surf, COLOR_OUTLINE, (tx, ty), BALL_RADIUS, 1)

        # Draw cue ball (white)
        cx, cy = int(self.cue_pos[0]), int(self.cue_pos[1])
        pygame.draw.circle(surf, COLOR_CUE,     (cx, cy), BALL_RADIUS)
        pygame.draw.circle(surf, COLOR_OUTLINE, (cx, cy), BALL_RADIUS, 1)

        # Copy to display surface if human mode
        if self.render_mode == "human" and self._screen is not None:
            self._screen.blit(surf, (0, 0))

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _random_position(rng, y_range=None):
        """Sample a position safely away from pocket corners."""
        margin = BALL_RADIUS + POCKET_RADIUS + 5
        x_range = (margin, WINDOW_SIZE - margin)
        y_lo, y_hi = y_range or (margin, WINDOW_SIZE - margin)
        y_lo = max(y_lo, margin)
        y_hi = min(y_hi, WINDOW_SIZE - margin)

        for _ in range(500):
            x = rng.uniform(x_range[0], x_range[1])
            y = rng.uniform(y_lo, y_hi)
            pos = np.array([x, y], dtype=np.float32)
            # Reject positions too close to pockets
            if all(np.linalg.norm(pos - p) > POCKET_RADIUS + BALL_RADIUS + 5
                   for p in POCKET_POSITIONS):
                return pos

        # Fallback: centre of the table
        return np.array([WINDOW_SIZE / 2, (y_lo + y_hi) / 2], dtype=np.float32)


# ── Standalone smoke-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running 10-step smoke test with random actions …")
    env = BilliardsEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        print(
            f"  step {i+1:2d} | obs={obs[:4].round(1)} | frame shape={frame.shape}")
        if terminated or truncated:
            print("  Episode ended early.")
            break
    env.close()
    print("Smoke test passed ✓")
