import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import pygame

DEPLOY_REAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEPLOY_REAL_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_REAL_DIR)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as OdometryGeo
from common.remote_controller import KeyMap
from keyboard.keyboard_command_dds import KeyboardCommand_, create_keyboard_command_message


class KeyboardDDSPublisher:
    def __init__(self, args):
        self.args = args

        pygame.init()
        pygame.display.set_caption("Keyboard DDS Controller")
        self._window = pygame.display.set_mode((980, 420))
        self._font = pygame.font.SysFont("monospace", 30)
        self._font_small = pygame.font.SysFont("monospace", 24)
        self._bg_color = (18, 18, 18)
        self._fg_color = (220, 220, 220)
        self._accent_color = (120, 220, 120)
        self._warn_color = (220, 180, 80)

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

        self._target_input_active = False
        self._target_position = np.asarray(args.initial_target, dtype=np.float32).ravel().copy()
        if self._target_position.shape[0] != 3:
            self._target_position = np.zeros(3, dtype=np.float32)
        self._pending_target_update = None
        self._target_error = ""
        self._target_labels = ["X", "Y", "Z"]
        self._target_text_fields = [f"{v:.3f}" for v in self._target_position]
        self._target_active_index = 0
        self._target_select_all = [False, False, False]
        self._last_click_time = 0.0
        self._last_click_index = -1
        self._double_click_interval_s = 0.35
        self._target_input_rects = [
            pygame.Rect(120 + i * 300, 165, 240, 58) for i in range(3)
        ]

        self._forward_input_active = False
        self._forward_distance_m = float(args.initial_forward_distance)
        self._forward_text_field = f"{self._forward_distance_m:.3f}"
        self._forward_select_all = False
        self._forward_input_rect = pygame.Rect(120, 295, 240, 58)
        self._forward_confirm_rect = pygame.Rect(390, 295, 360, 58)

        self._base_pose_lock = threading.Lock()
        self._base_pose_received = False
        self._base_pose_position = np.zeros(3, dtype=np.float32)
        self._base_pose_yaw = 0.0

        ChannelFactoryInitialize(0, args.net)
        self._pub = ChannelPublisher(args.topic, KeyboardCommand_)
        self._pub.Init()

        self._base_pose_sub = ChannelSubscriber(args.base_pose_topic, OdometryGeo)
        self._base_pose_sub.Init(self._base_pose_handler, 10)

        print(f"[KeyboardDDS] Publishing topic: {args.topic}")
        print(f"[KeyboardDDS] Base pose topic: {args.base_pose_topic}")
        print(
            "[KeyboardDDS] Controls: "
            "[w/s]=forward/back, [a/d]=left/right, [q/e]=yaw, "
            "[1]=start, [2]=A, [3]=select, [4/x]=X, [t]=edit target xyz, [f]=edit forward distance"
        )

    def _base_pose_handler(self, msg: OdometryGeo):
        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        pz = float(msg.pose.pose.position.z)

        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        with self._base_pose_lock:
            self._base_pose_position[:] = [px, py, pz]
            self._base_pose_yaw = yaw
            self._base_pose_received = True

    def _set_text_input_enabled(self):
        if self._target_input_active or self._forward_input_active:
            pygame.key.start_text_input()
        else:
            pygame.key.stop_text_input()

    def close(self):
        self._target_input_active = False
        self._forward_input_active = False
        self._set_text_input_enabled()
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
        if self._target_input_active:
            if keycode == pygame.K_RETURN:
                self._commit_target_fields()
            elif keycode == pygame.K_ESCAPE:
                self._cancel_target_input()
            elif keycode == pygame.K_BACKSPACE:
                if self._target_select_all[self._target_active_index]:
                    self._target_text_fields[self._target_active_index] = ""
                    self._target_select_all[self._target_active_index] = False
                else:
                    self._target_text_fields[self._target_active_index] = self._target_text_fields[
                        self._target_active_index
                    ][:-1]
            elif keycode == pygame.K_TAB:
                self._target_active_index = (self._target_active_index + 1) % 3
                self._target_select_all = [False, False, False]
            return

        if self._forward_input_active:
            if keycode == pygame.K_RETURN:
                self._set_target_in_front()
            elif keycode == pygame.K_ESCAPE:
                self._cancel_forward_input()
            elif keycode == pygame.K_BACKSPACE:
                if self._forward_select_all:
                    self._forward_text_field = ""
                    self._forward_select_all = False
                else:
                    self._forward_text_field = self._forward_text_field[:-1]
            return

        if keycode == pygame.K_1:
            self._set_pulse(KeyMap.start)
        elif keycode == pygame.K_2:
            self._set_pulse(KeyMap.A)
        elif keycode == pygame.K_3:
            self._set_pulse(KeyMap.select)
        elif keycode == pygame.K_4 or keycode == pygame.K_x:
            self._set_pulse(KeyMap.X)
        elif keycode == pygame.K_SPACE:
            for key in self._axis_active:
                self._axis_active[key] = False
        elif keycode == pygame.K_t:
            self._start_target_input(active_index=0)
        elif keycode == pygame.K_f:
            self._start_forward_input()

    def _on_text_input(self, text):
        if self._target_input_active:
            if text and all(ch in "0123456789-+eE." for ch in text):
                if self._target_select_all[self._target_active_index]:
                    self._target_text_fields[self._target_active_index] = text
                    self._target_select_all[self._target_active_index] = False
                else:
                    self._target_text_fields[self._target_active_index] += text
            return

        if self._forward_input_active:
            if text and all(ch in "0123456789-+eE." for ch in text):
                if self._forward_select_all:
                    self._forward_text_field = text
                    self._forward_select_all = False
                else:
                    self._forward_text_field += text
            return

    def _start_target_input(self, active_index):
        self._target_input_active = True
        self._forward_input_active = False
        self._target_active_index = int(np.clip(active_index, 0, 2))
        self._target_text_fields = [f"{v:.3f}" for v in self._target_position]
        self._target_select_all = [False, False, False]
        self._forward_select_all = False
        self._target_error = ""
        self._set_text_input_enabled()

    def _cancel_target_input(self):
        self._target_input_active = False
        self._target_select_all = [False, False, False]
        self._target_error = ""
        self._set_text_input_enabled()

    def _commit_target_fields(self):
        try:
            values = np.asarray([float(s) for s in self._target_text_fields], dtype=np.float32)
        except ValueError:
            self._target_error = "Invalid number format"
            return
        self._target_position = values
        self._pending_target_update = values.copy()
        self._target_input_active = False
        self._target_select_all = [False, False, False]
        self._target_error = ""
        self._set_text_input_enabled()

    def _start_forward_input(self):
        self._forward_input_active = True
        self._target_input_active = False
        self._forward_text_field = f"{self._forward_distance_m:.3f}"
        self._forward_select_all = False
        self._target_select_all = [False, False, False]
        self._target_error = ""
        self._set_text_input_enabled()

    def _cancel_forward_input(self):
        self._forward_input_active = False
        self._forward_select_all = False
        self._target_error = ""
        self._set_text_input_enabled()

    def _set_target_in_front(self):
        text = self._forward_text_field if self._forward_input_active else f"{self._forward_distance_m:.3f}"
        try:
            distance_m = float(text)
        except ValueError:
            self._target_error = "Invalid forward distance"
            return

        if distance_m < 0.0:
            self._target_error = "Forward distance must be >= 0"
            return

        with self._base_pose_lock:
            if not self._base_pose_received:
                self._target_error = "No base pose yet, cannot set front target"
                return
            x = float(self._base_pose_position[0])
            y = float(self._base_pose_position[1])
            yaw = float(self._base_pose_yaw)

        target = np.array([
            x + distance_m * math.cos(yaw),
            y + distance_m * math.sin(yaw),
            float(self._target_position[2]),
        ], dtype=np.float32)

        self._forward_distance_m = distance_m
        self._forward_text_field = f"{self._forward_distance_m:.3f}"
        self._target_position = target
        self._target_text_fields = [f"{v:.3f}" for v in self._target_position]
        self._pending_target_update = target.copy()
        self._forward_input_active = False
        self._forward_select_all = False
        self._target_error = ""
        self._set_text_input_enabled()

    def _on_mouse_buttondown(self, pos):
        if self._forward_confirm_rect.collidepoint(pos):
            self._set_target_in_front()
            return

        if self._forward_input_rect.collidepoint(pos):
            now = time.monotonic()
            is_double_click = (
                self._last_click_index == 3
                and (now - self._last_click_time) <= self._double_click_interval_s
            )
            self._last_click_index = 3
            self._last_click_time = now

            if not self._forward_input_active:
                self._start_forward_input()
            else:
                self._forward_select_all = False

            if is_double_click:
                self._forward_select_all = True
            return

        clicked_index = -1
        for i, rect in enumerate(self._target_input_rects):
            if rect.collidepoint(pos):
                clicked_index = i
                break
        if clicked_index < 0:
            return

        now = time.monotonic()
        is_double_click = (
            clicked_index == self._last_click_index
            and (now - self._last_click_time) <= self._double_click_interval_s
        )
        self._last_click_index = clicked_index
        self._last_click_time = now

        if not self._target_input_active:
            self._start_target_input(active_index=clicked_index)
        else:
            self._target_active_index = clicked_index
            self._target_select_all = [False, False, False]

        if is_double_click:
            self._target_select_all = [False, False, False]
            self._target_select_all[clicked_index] = True

    def _draw_ui(self):
        with self._base_pose_lock:
            base_pose_received = self._base_pose_received
            base_pose_x = float(self._base_pose_position[0])
            base_pose_y = float(self._base_pose_position[1])
            base_pose_yaw = float(self._base_pose_yaw)

        self._window.fill(self._bg_color)
        lines = [
            ("[w/s][a/d][q/e] move   [1][2][3] buttons   [4/x] nav toggle", self._fg_color),
            (
                f"Target xyz = ({self._target_position[0]: .3f}, {self._target_position[1]: .3f}, {self._target_position[2]: .3f})",
                self._accent_color,
            ),
            (
                f"BasePose = ({base_pose_x: .3f}, {base_pose_y: .3f}), yaw={base_pose_yaw: .3f} rad"
                if base_pose_received
                else "BasePose = waiting...",
                self._fg_color,
            ),
        ]
        if self._target_input_active:
            lines.append(("[Enter]=apply [Tab]=next box [Esc]=cancel", self._warn_color))
        elif self._forward_input_active:
            lines.append(("[Enter]=confirm front target [Esc]=cancel", self._warn_color))
        else:
            lines.append(("[t] edit target xyz | [f] edit front distance | or click input boxes", self._fg_color))

        if self._target_error:
            lines.append((self._target_error, (255, 120, 120)))

        y = 14
        for i, (text, color) in enumerate(lines):
            font = self._font if i < 2 else self._font_small
            surf = font.render(text, True, color)
            self._window.blit(surf, (12, y))
            y += 50 if i < 2 else 34

        for i, rect in enumerate(self._target_input_rects):
            active = self._target_input_active and i == self._target_active_index
            selected_all = active and self._target_select_all[i]
            border_color = self._accent_color if selected_all else (self._warn_color if active else (90, 90, 90))
            bg_color = (40, 40, 40) if active else (30, 30, 30)
            pygame.draw.rect(self._window, bg_color, rect, border_radius=8)
            pygame.draw.rect(self._window, border_color, rect, 3, border_radius=8)

            label = self._font_small.render(self._target_labels[i], True, self._fg_color)
            self._window.blit(label, (rect.x + 10, rect.y + 15))
            value_text = self._target_text_fields[i] if self._target_input_active else f"{self._target_position[i]:.3f}"
            value = self._font_small.render(value_text, True, self._accent_color)
            self._window.blit(value, (rect.x + 48, rect.y + 15))

        forward_active = self._forward_input_active
        forward_selected_all = self._forward_input_active and self._forward_select_all
        forward_border = self._accent_color if forward_selected_all else (self._warn_color if forward_active else (90, 90, 90))
        forward_bg = (40, 40, 40) if forward_active else (30, 30, 30)
        pygame.draw.rect(self._window, forward_bg, self._forward_input_rect, border_radius=8)
        pygame.draw.rect(self._window, forward_border, self._forward_input_rect, 3, border_radius=8)
        label = self._font_small.render("Front(m)", True, self._fg_color)
        self._window.blit(label, (self._forward_input_rect.x + 10, self._forward_input_rect.y + 15))
        forward_text = self._forward_text_field if self._forward_input_active else f"{self._forward_distance_m:.3f}"
        forward_value = self._font_small.render(forward_text, True, self._accent_color)
        self._window.blit(forward_value, (self._forward_input_rect.x + 112, self._forward_input_rect.y + 15))

        btn_bg = (36, 72, 36)
        btn_border = self._accent_color
        pygame.draw.rect(self._window, btn_bg, self._forward_confirm_rect, border_radius=8)
        pygame.draw.rect(self._window, btn_border, self._forward_confirm_rect, 3, border_radius=8)
        btn_text = self._font_small.render("Set Target In Front", True, self._fg_color)
        self._window.blit(btn_text, (self._forward_confirm_rect.x + 52, self._forward_confirm_rect.y + 15))
        pygame.display.flip()

    def run(self):
        publish_dt = 1.0 / max(self.args.publish_hz, 1e-6)
        next_pub = time.monotonic()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        self._on_keydown(event.key)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._on_mouse_buttondown(event.pos)
                    elif event.type == pygame.TEXTINPUT:
                        self._on_text_input(event.text)

                now = time.monotonic()
                pressed = pygame.key.get_pressed()
                if not self._target_input_active and not self._forward_input_active:
                    self._update_axis_state(now, pressed)
                else:
                    for key in self._axis_active:
                        self._axis_active[key] = False

                lx = self._harmonic_gain("d", now) - self._harmonic_gain("a", now)
                ly = self._harmonic_gain("w", now) - self._harmonic_gain("s", now)
                rx = self._harmonic_gain("e", now) - self._harmonic_gain("q", now)

                if now >= next_pub:
                    next_pub += publish_dt
                    target_update = self._pending_target_update
                    msg = create_keyboard_command_message(
                        buttons=self._button_pulse,
                        lx=self._clamp(lx),
                        ly=self._clamp(ly),
                        rx=self._clamp(rx),
                        ry=0.0,
                        target_world=target_update,
                    )
                    self._pub.Write(msg)
                    self._button_pulse = [0] * 16
                    self._pending_target_update = None

                self._draw_ui()
                time.sleep(0.001)
        finally:
            self.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Publish pygame keyboard commands via DDS")
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("--topic", type=str, default="rt/wireless_remote", help="DDS topic for keyboard remote command")
    parser.add_argument("--base-pose-topic", type=str, default="rt/base_pose", help="DDS unified base pose topic used to set forward target")
    parser.add_argument("--publish-hz", type=float, default=50.0, help="DDS publish rate")
    parser.add_argument("--initial-forward-distance", type=float, default=1.0, help="default forward distance in meters")
    parser.add_argument(
        "--initial-target",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.5],
        metavar=("X", "Y", "Z"),
        help="initial target position used by keyboard UI",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = KeyboardDDSPublisher(args)
    app.run()


if __name__ == "__main__":
    main()
