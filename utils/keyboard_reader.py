import threading
import time

import numpy as np
import pygame
from pygame.locals import *


class KeyboardController:
    """Keyboard controller that reads from keyboard using pygame with first-order filter."""

    def __init__(
        self,
        vel_scale_x=0.4,
        vel_scale_y=0.4,
        vel_scale_rot=1.0,
        filter_alpha=0.1,  # Filter coefficient: 0 < alpha < 1, smaller = smoother
    ):
        self._vel_scale_x = vel_scale_x
        self._vel_scale_y = vel_scale_y
        self._vel_scale_rot = vel_scale_rot
        self._filter_alpha = filter_alpha

        # Target velocities (what we're moving towards)
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_wz = 0.0

        # Filtered velocities (smoothed output)
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.is_running = True

        # Initialize pygame
        pygame.init()
        pygame.mixer.quit()
        # Create a small window to capture keyboard events
        self._screen = pygame.display.set_mode((200, 100))
        pygame.display.set_caption("Keyboard Controller (Focus here)")

        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()

    def read_loop(self):
        print("Keyboard controller started. Use WASD to move, Q/E to rotate.")
        print("Press ESC to quit.")

        while self.is_running:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                    return

            # Get pressed keys
            keys = pygame.key.get_pressed()

            # Check for ESC to quit
            if keys[pygame.K_ESCAPE]:
                self.is_running = False
                return

            # Update velocities based on key presses
            self.update_command(keys)

            # Small sleep to avoid high CPU usage
            time.sleep(0.01)

        pygame.quit()

    def update_command(self, keys):
        # Set target velocities based on key presses
        target_vx = 0.0
        target_vy = 0.0
        target_wz = 0.0

        # Forward/Backward (W/S)
        if keys[pygame.K_w]:
            target_vx = self._vel_scale_x
        if keys[pygame.K_s]:
            target_vx = -self._vel_scale_x

        # Left/Right strafe (A/D)
        if keys[pygame.K_a]:
            target_vy = self._vel_scale_y
        if keys[pygame.K_d]:
            target_vy = -self._vel_scale_y

        # Rotation (Q/E)
        if keys[pygame.K_q]:
            target_wz = self._vel_scale_rot
        if keys[pygame.K_e]:
            target_wz = -self._vel_scale_rot

        # Update targets
        self._target_vx = target_vx
        self._target_vy = target_vy
        self._target_wz = target_wz

        # Apply first-order filter: y = y + alpha * (target - y)
        self.vx = self.vx + self._filter_alpha * (self._target_vx - self.vx)
        self.vy = self.vy + self._filter_alpha * (self._target_vy - self.vy)
        self.wz = self.wz + self._filter_alpha * (self._target_wz - self.wz)

        # Apply small threshold to avoid floating point noise
        if abs(self.vx) < 1e-4:
            self.vx = 0.0
        if abs(self.vy) < 1e-4:
            self.vy = 0.0
        if abs(self.wz) < 1e-4:
            self.wz = 0.0

    def get_command(self):
        return np.array([self.vx, self.vy, self.wz])

    def stop(self):
        self.is_running = False
        pygame.quit()


if __name__ == "__main__":
    # Test keyboard controller
    print("Testing keyboard controller...")
    print("Focus on the pygame window and use WASD + Q/E to control.")
    controller = KeyboardController(filter_alpha=0.05)
    try:
        while controller.is_running:
            cmd = controller.get_command()
            # Use \r to overwrite the same line, end="" to avoid newline
            print(
                f"\rCommand: vx={cmd[0]:+.3f}, vy={cmd[1]:+.3f}, wz={cmd[2]:+.3f}    ",
                end="",
                flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()  # Print newline before exiting
        controller.stop()
