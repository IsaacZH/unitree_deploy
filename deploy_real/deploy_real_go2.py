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

    # def zero_torque_state(self):
    #     print("Enter zero torque state.")
    #     print("Waiting for the start signal...")
    #     while self.remote_controller.button[KeyMap.start] != 1:
    #         create_zero_cmd(self.low_cmd)
    #         self.send_cmd(self.low_cmd)
    #         time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)
        kps = self.config.fixstand_kp
        kds = self.config.fixstand_kd
        ts = self.config.fixstand_ts
        qs = self.config.fixstand_qs
        dof_size = len(self.config.joint_ids_map)

        # Capture current position as first waypoint
        q0 = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            q0[i] = self.low_state.motor_state[i].q
        print("Current position captured as first waypoint:", q0)
        qs[0] = q0.tolist()

        t0 = time.time()

        while True:
            t = time.time() - t0
            q = self._linear_interpolate(t, ts, qs)

            for j in range(dof_size):
                self.low_cmd.motor_cmd[j].q = q[j]
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = kps[j]
                self.low_cmd.motor_cmd[j].kd = kds[j]
                self.low_cmd.motor_cmd[j].tau = 0
            self.send_cmd(self.low_cmd)

            if t >= ts[-1]:
                break
            time.sleep(self.config.control_dt)

        while self.remote_controller.button[KeyMap.A] != 1:
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _linear_interpolate(self, t, ts, qs):
        if t <= ts[0]:
            return np.array(qs[0], dtype=np.float32)
        if t >= ts[-1]:
            return np.array(qs[-1], dtype=np.float32)

        for i in range(len(ts) - 1):
            if t >= ts[i] and t <= ts[i + 1]:
                alpha = (t - ts[i]) / (ts[i + 1] - ts[i])
                result = []
                for j in range(len(qs[i])):
                    result.append(qs[i][j] * (1 - alpha) + qs[i + 1][j] * alpha)
                return np.array(result, dtype=np.float32)

        return np.array(qs[-1], dtype=np.float32)

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
    parser.add_argument("--mujoco", action="store_true", help="use mujoco simulation")
    args = parser.parse_args()

    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config, use_mujoco=args.mujoco)

    if not args.mujoco:
        controller.shut_down_control_service()
        # controller.zero_torque_state()


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
