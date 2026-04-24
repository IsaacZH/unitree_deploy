import numpy as np
import time
import torch
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_go
from common.remote_controller import RemoteController, KeyMap
from config import Config
from terms import TERMS
from observation_manager import ObservationManager, PolicyInputManager
from action_manager import ActionManager
from state_machine import StateMachine


class Controller:
    def __init__(self, config: Config, use_mujoco: bool = False) -> None:
        self.config = config
        self.use_mujoco = use_mujoco
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

        self.target_dof_pos = config.default_joint_pos.copy()

        
        if not use_mujoco:
            self.sc = SportClient()
            self.sc.SetTimeout(5.0)
            self.sc.Init()
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        if not use_mujoco:
            self.wait_for_low_state()

        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.obs_manager.reset()

        # Initialize state machine
        self.state_machine = StateMachine(self, initial_state_name="move_to_default_pos")

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

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
    args = parser.parse_args()

    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config, use_mujoco=args.mujoco)

    if not args.mujoco:
        controller.shut_down_control_service()

    # Run state machine
    while controller.state_machine.execute():
        pass

    print("Exit")
