import argparse
import os
import sys
import time
import numpy as np
import torch

DEPLOY_REAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEPLOY_REAL_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_REAL_DIR)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowState_,
    unitree_go_msg_dds__SportModeState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as SportModeStateGo

from config import Config
from terms import TERMS
from common.depth_image_sub import DepthImageObserver
from common.navigation_command_manager import NavigationCommandManager
from common.nav_debug_dds import NavDebug_, create_nav_debug_message
from common.nav_target_dds import NavTarget_
from common.raw_state import RawState
from common.remote_controller import RemoteController


class NavDryRunRunner:
    def __init__(self, config: Config, use_mujoco: bool = False):
        self.config = config
        self.use_mujoco = use_mujoco
        self.remote_controller = RemoteController()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.last_action = np.zeros(len(config.default_joint_pos), dtype=np.float32)
        self.nav_last_action = np.zeros(3, dtype=np.float32)

        nav_cfg = dict(config.navigation)
        nav_cfg["enabled"] = True
        self.navigation_manager = NavigationCommandManager(nav_cfg, TERMS)
        self.navigation_manager.control_source = "navigation"

        dc = config.depth_camera
        self.depth_observer = DepthImageObserver(
            topic=dc["topic"],
            min_depth=dc["min_depth"],
            max_depth=dc["max_depth"],
            target_resolution=dc["resolution"],
            encoder_path=dc["encoder_path"],
            feature_dim=dc["feature_dim"],
            device=dc["device"],
            enable_noise=dc.get("enable_noise", False),
            focal_length=dc.get("focal_length", None),
            baseline=dc.get("baseline", None),
            use_jit_precompiled=dc.get("use_jit_precompiled", False),
            visualize_depth=dc.get("visualize_depth", False),
        )

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.highstate_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeStateGo)
        self.highstate_subscriber.Init(self.high_state_handler, 10)
        self.nav_target_subscriber = ChannelSubscriber("rt/nav_target", NavTarget_)
        self.nav_target_subscriber.Init(self.nav_target_handler, 10)
        self.nav_debug_publisher = ChannelPublisher("rt/nav_debug", NavDebug_)
        self.nav_debug_publisher.Init()

    def low_state_handler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def high_state_handler(self, msg: SportModeStateGo):
        self.high_state = msg

    def nav_target_handler(self, msg: NavTarget_):
        target = np.asarray(msg.target_world, dtype=np.float32).ravel()
        if target.shape[0] >= 3:
            self.navigation_manager.set_target_position(target[:3])

    def build_raw_state(self) -> RawState:
        return RawState(
            remote=self.remote_controller,
            imu=self.low_state.imu_state,
            motor_state=self.low_state.motor_state,
            high_state=self.high_state,
            navigation_manager=self.navigation_manager,
            last_action=self.last_action,
            nav_last_action=self.nav_last_action,
            depth_feature=self.depth_observer.get_latest(),
        )

    def wait_for_low_state(self, timeout_s: float = 5.0):
        start = time.monotonic()
        while self.low_state.tick == 0:
            if time.monotonic() - start > timeout_s:
                print("[NavDryRun] LowState wait timeout, continue without blocking.")
                return
            time.sleep(self.config.control_dt)
        print("[NavDryRun] LowState connected.")

    def get_robot_position(self) -> np.ndarray:
        if hasattr(self.high_state, "position"):
            pos = np.asarray(self.high_state.position, dtype=np.float32).ravel()
            if pos.shape[0] >= 3:
                return pos[:3]
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    def run(self, print_hz: float = 10.0):
        if not self.use_mujoco:
            self.wait_for_low_state()
        self.navigation_manager.reset_policy()
        print("[NavDryRun] Running navigation dry-run (no control publish).")

        print_period = 1.0 / print_hz
        next_print_time = time.monotonic()

        while True:
            raw_state = self.build_raw_state()
            cmd = self.navigation_manager.get_navigation_velocity_command(raw_state, self.config)
            self.nav_last_action = np.asarray(cmd, dtype=np.float32).ravel()

            robot_pos = self.get_robot_position()
            target_pos_world = self.navigation_manager.get_target_position()
            target_obs = TERMS["target_position"](raw_state, self.config)
            target_dir = np.asarray(target_obs, dtype=np.float32).ravel()[:3]
            base_lin_obs = np.asarray(TERMS["base_lin_vel"](raw_state, self.config), dtype=np.float32).ravel()
            base_ang_obs = np.asarray(TERMS["base_ang_vel"](raw_state, self.config), dtype=np.float32).ravel()
            msg = create_nav_debug_message(target_dir_b=target_dir, target_speed_b=cmd)
            self.nav_debug_publisher.Write(msg)

            now = time.monotonic()
            if now >= next_print_time:
                target_pos = target_pos_world
                print(
                    f"[NavDryRun] t={time.time():.3f} source=navigation "
                    f"vx={cmd[0]: .3f} vy={cmd[1]: .3f} wz={cmd[2]: .3f} "
                    f"v_obs=({base_lin_obs[0]: .3f},{base_lin_obs[1]: .3f},{base_lin_obs[2]: .3f}) "
                    f"w_obs=({base_ang_obs[0]: .3f},{base_ang_obs[1]: .3f},{base_ang_obs[2]: .3f}) "
                    f"robot=({robot_pos[0]: .3f},{robot_pos[1]: .3f},{robot_pos[2]: .3f}) "
                    f"target=({target_pos[0]: .3f},{target_pos[1]: .3f},{target_pos[2]: .3f})"
                )
                next_print_time = now + print_period

            time.sleep(self.config.control_dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder")
    parser.add_argument("--mujoco", action="store_true", help="kept for CLI compatibility")
    parser.add_argument("--print-hz", type=float, default=10.0, help="velocity command print frequency")
    args = parser.parse_args()

    config = Config(f"deploy_real/configs/{args.config}")
    if config.depth_camera is None:
        raise ValueError("depth_camera must be configured for navigation dry-run.")

    ChannelFactoryInitialize(0, args.net)
    torch.set_grad_enabled(False)

    runner = NavDryRunRunner(config, use_mujoco=args.mujoco)
    try:
        runner.run(print_hz=args.print_hz)
    except KeyboardInterrupt:
        print("\n[NavDryRun] Exit.")
