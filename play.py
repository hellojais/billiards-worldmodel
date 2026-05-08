"""
play.py - Interactive Human Play Mode

Lets a human control the cue ball with the mouse:
  - Left-click to set the direction and shoot.
  - The cue ball receives an impulse toward the click point.
  - No data is recorded — this is purely for visual inspection / fun.

Controls
--------
    Left-click  : Shoot cue ball toward mouse cursor
    R           : Reset episode
    ESC / Q     : Quit
"""

import sys
import pygame
import numpy as np
from game import BilliardsEnv, WINDOW_SIZE, COLOR_FELT

IMPULSE_STRENGTH = 18.0  # pixels/step — shot strength


def _balls_moving(obs: np.ndarray, threshold: float = 0.3) -> bool:
    """Return True if either ball has velocity above *threshold*."""
    cue_speed = np.linalg.norm(obs[2:4])
    target_speed = np.linalg.norm(obs[6:8])
    return cue_speed > threshold or target_speed > threshold


def run():
    # Use rgb_array so env.render() never calls display.flip() internally.
    # We blit the surface manually and do exactly ONE flip per frame.
    env = BilliardsEnv(render_mode="rgb_array")
    obs, _ = env.reset()

    # Initialise pygame display ourselves (env won't open a window in rgb_array mode)
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Billiards")
    last_caption = "Billiards"

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    step = 0
    done = False
    status = "Shoot! (left-click)"

    # shot_pending: True the frame after the player clicks, so the step fires immediately
    shot_pending = False
    pending_action = np.zeros(2, dtype=np.float32)

    while True:
        # ── Event handling ────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    env.close()
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    obs, _ = env.reset()
                    step = 0
                    done = False
                    shot_pending = False
                    pending_action = np.zeros(2, dtype=np.float32)
                    status = "Shoot! (left-click)"

            # Accept a click whenever balls are still and game is live
            if (event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and not done
                    and not _balls_moving(obs)):
                mx, my = pygame.mouse.get_pos()
                cue_pos = obs[0:2]
                direction = np.array(
                    [mx - cue_pos[0], my - cue_pos[1]], dtype=np.float32)
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    pending_action = direction / dist * IMPULSE_STRENGTH
                    shot_pending = True

        # ── Advance physics when rolling OR immediately after a click ─────────
        if not done and (_balls_moving(obs) or shot_pending):
            obs, reward, terminated, truncated, _ = env.step(pending_action)
            pending_action = np.zeros(2, dtype=np.float32)
            shot_pending = False
            step += 1
            done = terminated or truncated

            if terminated:
                status = f"★ POTTED in {step} steps!  Press R to replay."
            elif truncated:
                status = f"Time up after {step} steps. Press R to retry."
            else:
                status = f"Step {step}/300  —  left-click to shoot"

        # ── Update window caption only when status text actually changes ───────
        caption = f"Billiards  |  {status}"
        if caption != last_caption:
            pygame.display.set_caption(caption)
            last_caption = caption

        # ── Render game scene (no flip inside), blit to window ────────────────
        env.render()                        # draws onto env._surface only
        screen.blit(env._surface, (0, 0))  # copy game to display

        # Aim line + crosshair while waiting for a shot
        if not done and not _balls_moving(obs):
            mx, my = pygame.mouse.get_pos()
            cx, cy = int(obs[0]), int(obs[1])
            pygame.draw.line(screen, (255, 255, 100), (cx, cy), (mx, my), 1)
            pygame.draw.line(screen, (255, 255, 100),
                             (mx - 8, my), (mx + 8, my), 1)
            pygame.draw.line(screen, (255, 255, 100),
                             (mx, my - 8), (mx, my + 8), 1)

        # Stable opaque HUD bar drawn once on top of everything
        bar = pygame.Surface((WINDOW_SIZE, 44), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 170))
        screen.blit(bar, (0, 0))
        screen.blit(font.render(status, True, (255, 255, 180)), (8, 6))
        screen.blit(font.render(
            "Click on/past the RED ball  |  R = reset  |  Q = quit",
            True, (180, 230, 180)), (8, 24))

        pygame.display.flip()   # exactly ONE flip per frame
        clock.tick(60)


if __name__ == "__main__":
    run()
