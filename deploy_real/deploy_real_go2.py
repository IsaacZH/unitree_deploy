import time
import numpy as np
import torch
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_go_msg_dds__SportModeState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as SportModeStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from common.command_helper import init_cmd_go
from common.remote_controller import RemoteController
from common.depth_image_sub import DepthImageObserver
from common.keyboard_controller import KeyboardController
from common.navigation_command_manager import NavigationCommandManager
from common.nav_debug_dds import NavDebug_, create_nav_debug_message
from common.nav_target_dds import NavTarget_, create_nav_target_message
from config import Config
from terms import TERMS
from observation_manager import ObservationManager, PolicyInputManager
from action_manager import ActionManager
from state_machine import StateMachine


class Controller:
    def __init__(self, config: Config, use_mujoco: bool = False, keyboard: bool = False) -> None:
        self.config = config
        self.use_mujoco = use_mujoco
        self.keyboard_mode = keyboard
        self.remote_controller = RemoteController()

        self.policy = torch.jit.load(config.policy_path)
        self.actor = self.policy.actor
        self.encoder = self.policy.estimator

        self.obs_manager = ObservationManager(config, TERMS, self.encoder)
        self.policy_input_manager = PolicyInputManager(
            config,
            self.obs_manager.obs_offsets,
            self.obs_manager.obs_sizes,
            self.obs_manager.obs_names
        )
        self.action_manager = ActionManager(config)
        self.nav_last_action = np.zeros(3, dtype=np.float32)

        self.target_dof_pos = config.default_joint_pos.copy()
        self.navigation_manager = NavigationCommandManager(config.navigation, TERMS)
        self.keyboard_controller = (
            KeyboardController(initial_target_position=self.navigation_manager.get_target_position())
            if self.keyboard_mode
            else None
        )

        
        if not use_mujoco:
            self.sc = SportClient()
            self.sc.SetTimeout(5.0)
            self.sc.Init()
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.high_state = unitree_go_msg_dds__SportModeState_()

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdGo)
        self.lowcmd_publisher_.Init()
        self.nav_debug_publisher_ = ChannelPublisher("rt/nav_debug", NavDebug_)
        self.nav_debug_publisher_.Init()
        self.nav_target_publisher_ = ChannelPublisher("rt/nav_target", NavTarget_)
        self.nav_target_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        self.highstate_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeStateGo)
        self.highstate_subscriber.Init(self.HighStateGoHandler, 10)

        if not use_mujoco:
            self.wait_for_low_state()

        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.obs_manager.reset()

        # Initialize depth camera observer (optional)
        if config.depth_camera is not None:
            dc = config.depth_camera
            self.depth_observer = DepthImageObserver(
                topic=dc["topic"],
                min_depth=dc["min_depth"],
                max_depth=dc["max_depth"],
                target_resolution=dc["resolution"],
                encoder_path=dc["encoder_path"],
                feature_dim=dc["feature_dim"],
                device=dc["device"],
            )
        else:
            self.depth_observer = None

        # Initialize state machine
        self.state_machine = StateMachine(self, initial_state_name="move_to_default_pos")
        self.publish_target_update(self.navigation_manager.get_target_position())

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        if not self.keyboard_mode:
            self.remote_controller.set(self.low_state.wireless_remote)

    def HighStateGoHandler(self, msg: SportModeStateGo):
        self.high_state = msg

    def update_control_input(self):
        if self.keyboard_controller is not None:
            self.keyboard_controller.update_remote(self.remote_controller)
            target_update = self.keyboard_controller.consume_target_position_update()
            if target_update is not None:
                self.navigation_manager.set_target_position(target_update)
                self.publish_target_update(target_update)
                print(
                    "[Navigation] Target updated from keyboard: "
                    f"({target_update[0]:.3f}, {target_update[1]:.3f}, {target_update[2]:.3f})"
                )
        self.navigation_manager.update_control_source(self.remote_controller.button)

    def publish_target_update(self, target_world):
        msg = create_nav_target_message(np.asarray(target_world, dtype=np.float32).ravel())
        self.nav_target_publisher_.Write(msg)

    def publish_nav_debug(self, raw_state):
        target_obs = TERMS["target_position"](raw_state, self.config)
        target_dir = np.asarray(target_obs, dtype=np.float32).ravel()[:3]
        cmd = getattr(raw_state, "velocity_command", None)
        if cmd is None:
            cmd = TERMS["velocity_commands"](raw_state, self.config)
        msg = create_nav_debug_message(target_dir_b=target_dir, target_speed_b=cmd)
        self.nav_debug_publisher_.Write(msg)

    def close(self):
        if self.keyboard_controller is not None:
            self.keyboard_controller.close()

    def send_cmd(self, cmd: LowCmdGo):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def shut_down_control_service(self):
        _, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            _, result = self.msc.CheckMode()
            time.sleep(1)
        print("Successfully shut down the operation control service.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder")
    parser.add_argument("--mujoco", action="store_true", help="use mujoco simulation")
    parser.add_argument("--keyboard", action="store_true", help="use keyboard for state transitions and velocity commands")
    args = parser.parse_args()

    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config, use_mujoco=args.mujoco, keyboard=args.keyboard)

    if not args.mujoco:
        controller.shut_down_control_service()

    # Run state machine
    try:
        while controller.state_machine.execute():
            pass
    finally:
        controller.close()

    print("Exit")
