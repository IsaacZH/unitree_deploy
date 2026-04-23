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


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
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

        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        self.wait_for_low_state()

        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.obs_manager.reset()

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

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.joint_ids_map
        kps = self.config.stiffness
        kds = self.config.damping
        default_pos = self.config.default_joint_pos
        dof_size = len(dof_idx)

        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
            
        while self.remote_controller.button[KeyMap.A] != 1:
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _build_raw_state(self):
        class RawState:
            pass
        raw = RawState()
        raw.remote = self.remote_controller
        raw.imu = self.low_state.imu_state
        raw.motor_state = self.low_state.motor_state
        raw.last_action = self.obs_manager.last_action
        return raw

    def run(self):
        raw_state = self._build_raw_state()

        current_obs, encoder_output = self.obs_manager.forward(raw_state)

        policy_obs = self.policy_input_manager.forward(current_obs, encoder_output)
        obs_tensor = torch.from_numpy(policy_obs.copy()).unsqueeze(0)

        action = self.actor(obs_tensor).detach().numpy().squeeze()

        target_dof_pos = self.action_manager.forward(action)

        for i, motor_idx in enumerate(self.config.joint_ids_map):
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        self.obs_manager.set_last_action(action)
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    controller.shut_down_control_service()

    controller.zero_torque_state()

    controller.move_to_default_pos()

    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
