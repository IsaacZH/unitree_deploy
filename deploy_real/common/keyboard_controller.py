import time
import math
import pygame

from common.remote_controller import KeyMap


class KeyboardController:
    """Terminal keyboard input for RemoteController-compatible commands."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Keyboard Controller")
        self._window = pygame.display.set_mode((480, 120))

        self._harmonic_ramp_time = 0.6
        self._axis_limit = 1.0
        self._axis_keycode = {
            "w": pygame.K_w,
            "s": pygame.K_s,
            "a": pygame.K_a,
            "d": pygame.K_d,
            "q": pygame.K_q,
            "e": pygame.K_e,
        }
        self._axis_active = {k: False for k in self._axis_keycode}
        self._axis_press_start = {k: 0.0 for k in self._axis_keycode}
        self._button_pulse = [0] * 16

        print(
            "Keyboard control enabled: "
            "[w/s]=forward/back, [a/d]=left/right, [q/e]=yaw, "
            "[1]=start, [2]=A, [3]=select. "
            "Keep the 'Keyboard Controller' window focused."
        )

    def close(self):
        pygame.quit()

    def _set_pulse(self, key_id: int):
        self._button_pulse[key_id] = 1

    def _update_axis_state(self, now, pressed):
        for key, keycode in self._axis_keycode.items():
            active_now = bool(pressed[keycode])
            if active_now and not self._axis_active[key]:
                self._axis_active[key] = True
                self._axis_press_start[key] = now
            elif not active_now and self._axis_active[key]:
                self._axis_active[key] = False

    def _harmonic_gain(self, key, now):
        if not self._axis_active[key]:
            return 0.0
        t = max(0.0, now - self._axis_press_start[key])
        phase = min(1.0, t / self._harmonic_ramp_time)
        return math.sin(0.5 * math.pi * phase)

    def _clamp(self, value):
        return max(-self._axis_limit, min(self._axis_limit, value))

    def _on_keydown(self, keycode):
        if keycode == pygame.K_1:
            self._set_pulse(KeyMap.start)
        elif keycode == pygame.K_2:
            self._set_pulse(KeyMap.A)
        elif keycode == pygame.K_3:
            self._set_pulse(KeyMap.select)
        elif keycode == pygame.K_SPACE:
            for key in self._axis_active:
                self._axis_active[key] = False

    def update_remote(self, remote):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self._on_keydown(event.key)

        now = time.monotonic()
        pressed = pygame.key.get_pressed()
        self._update_axis_state(now, pressed)

        lx = self._harmonic_gain("d", now) - self._harmonic_gain("a", now)
        ly = self._harmonic_gain("w", now) - self._harmonic_gain("s", now)
        rx = self._harmonic_gain("e", now) - self._harmonic_gain("q", now)

        remote.lx = float(self._clamp(lx))
        remote.ly = float(self._clamp(ly))
        remote.rx = float(self._clamp(rx))
        remote.ry = 0.0
        for i in range(16):
            remote.button[i] = self._button_pulse[i]
        self._button_pulse = [0] * 16
