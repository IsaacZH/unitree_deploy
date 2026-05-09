import time
import numpy as np
import torch
import threading
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_go_msg_dds__SportModeState_,
    nav_msgs_msg_dds__Odometry_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as SportModeStateGo
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as OdometryGeo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from common.command_helper import init_cmd_go
from common.remote_controller import RemoteController
from common.depth_image_sub import DepthImageObserver
from common.navigation_command_manager import NavigationCommandManager
from common.nav_debug_dds import NavDebug_, create_nav_debug_message
from common.nav_target_dds import NavTarget_, create_nav_target_message
from keyboard.keyboard_command_dds import KeyboardCommand_
from config import Config
from terms import TERMS
from observation_manager import ObservationManager, PolicyInputManager
from action_manager import ActionManager
from state_machine import StateMachine


class Controller:
    class _HighStateProxy:
        def __init__(self):
            self.position = np.zeros(3, dtype=np.float32)
            self.velocity = np.zeros(3, dtype=np.float32)

    def __init__(
        self,
        config: Config,
        use_mujoco: bool = False,
        keyboard: bool = False,
        keyboard_topic: str = "rt/wireless_remote",
        base_pose_topic: str = "rt/base_pose",
    ) -> None:
        self.config = config
        self.use_mujoco = use_mujoco
        self.keyboard_mode = keyboard
        self.keyboard_topic = keyboard_topic
        self.base_pose_topic = base_pose_topic
        self.remote_controller = RemoteController()
        self._keyboard_lock = threading.Lock()
        self._keyboard_rx_count = 0
        self._last_lowstate_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

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
        self.keyboard_controller = None

        
        if not use_mujoco:
            self.sc = SportClient()
            self.sc.SetTimeout(5.0)
            self.sc.Init()
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.high_state = unitree_go_msg_dds__SportModeState_() if use_mujoco else self._HighStateProxy()
        self.inekf_odom = nav_msgs_msg_dds__Odometry_()

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdGo)
        self.lowcmd_publisher_.Init()
        self.nav_debug_publisher_ = ChannelPublisher("rt/nav_debug", NavDebug_)
        self.nav_debug_publisher_.Init()
        self.nav_target_publisher_ = ChannelPublisher("rt/nav_target", NavTarget_)
        self.nav_target_publisher_.Init()
        self.base_pose_publisher_ = ChannelPublisher(self.base_pose_topic, OdometryGeo)
        self.base_pose_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        
        # Always subscribe to keyboard topic for target position updates
        self.keyboard_subscriber = ChannelSubscriber(self.keyboard_topic, KeyboardCommand_)
        self.keyboard_subscriber.Init(self.KeyboardCmdHandler, 10)
        if self.keyboard_mode:
            print(f"[Controller] Keyboard DDS enabled (full control), subscribed: {self.keyboard_topic}")
        else:
            print(f"[Controller] Keyboard DDS enabled (target position only), subscribed: {self.keyboard_topic}")
        print(f"[Controller] Unified base pose publishing: {self.base_pose_topic}")
        if use_mujoco:
            self.highstate_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeStateGo)
            self.highstate_subscriber.Init(self.HighStateGoHandler, 10)
        else:
            self.odom_subscriber = ChannelSubscriber("rt/inekf/odom", OdometryGeo)
            self.odom_subscriber.Init(self.OdomHandler, 10)

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
                enable_noise=dc.get("enable_noise", False),
                focal_length=dc.get("focal_length", None),
                baseline=dc.get("baseline", None),
                use_jit_precompiled=dc.get("use_jit_precompiled", False),
                use_amp=dc.get("use_amp", False),
                compile_encoder=dc.get("compile_encoder", False),
                visualize_depth=dc.get("visualize_depth", False),
                visualize_topic=dc.get("visualize_topic", "rt/depth_image_noisy"),
            )
        else:
            self.depth_observer = None

        # Initialize state machine
        self.state_machine = StateMachine(self, initial_state_name="move_to_default_pos")
        self.publish_target_update(self.navigation_manager.get_target_position())

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        try:
            q = np.asarray(msg.imu_state.quaternion, dtype=np.float32).ravel()
            if q.shape[0] >= 4:
                # Unitree IMU quaternion is wxyz; Odometry uses xyzw.
                quat = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
                norm = float(np.linalg.norm(quat))
                if norm > 1e-6:
                    self._last_lowstate_quat_xyzw = (quat / norm).astype(np.float32)
        except Exception:
            pass
        if not self.keyboard_mode:
            self.remote_controller.set(self.low_state.wireless_remote)

    def HighStateGoHandler(self, msg: SportModeStateGo):
        self.high_state = msg
        self._publish_base_pose_from_sport(msg)

    def KeyboardCmdHandler(self, msg: KeyboardCommand_):
        # Always handle target position update (regardless of keyboard mode)
        target_update = None
        try:
            if bool(getattr(msg, "has_target_update", False)):
                target = np.asarray(msg.target_world, dtype=np.float32).ravel()
                if target.shape[0] >= 3:
                    target_update = target[:3].copy()
        except Exception:
            pass
        
        if target_update is not None:
            self.navigation_manager.set_target_position(target_update)
            self.publish_target_update(target_update)
            print(
                "[Navigation] Target updated from keyboard DDS: "
                f"({target_update[0]:.3f}, {target_update[1]:.3f}, {target_update[2]:.3f})"
            )
        
        # Handle remote controller input only in keyboard mode
        if self.keyboard_mode:
            buttons = [int(v) for v in msg.buttons[:16]]
            with self._keyboard_lock:
                self.remote_controller.set_axes(
                    lx=float(msg.lx),
                    ly=float(msg.ly),
                    rx=float(msg.rx),
                    ry=float(msg.ry),
                )
                self.remote_controller.set_buttons(buttons)
                self._keyboard_rx_count += 1

    def OdomHandler(self, msg: OdometryGeo):
        self.inekf_odom = msg
        self.high_state.position = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=np.float32,
        )
        self.high_state.velocity = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            dtype=np.float32,
        )
        self._publish_base_pose_from_odom(msg)

    def _make_base_pose_msg(
        self,
        position_xyz: np.ndarray,
        velocity_xyz: np.ndarray,
        quat_xyzw: np.ndarray,
    ):
        msg = nav_msgs_msg_dds__Odometry_()
        now = time.time()
        stamp_ns = int(now * 1e9)
        msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
        msg.header.frame_id = "world"
        msg.child_frame_id = "base"

        msg.pose.pose.position.x = float(position_xyz[0])
        msg.pose.pose.position.y = float(position_xyz[1])
        msg.pose.pose.position.z = float(position_xyz[2])

        msg.pose.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.pose.orientation.w = float(quat_xyzw[3])

        msg.twist.twist.linear.x = float(velocity_xyz[0])
        msg.twist.twist.linear.y = float(velocity_xyz[1])
        msg.twist.twist.linear.z = float(velocity_xyz[2])
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = 0.0
        return msg

    def _publish_base_pose_from_odom(self, msg: OdometryGeo):
        self.base_pose_publisher_.Write(msg)

    def _publish_base_pose_from_sport(self, msg: SportModeStateGo):
        position = np.asarray(getattr(msg, "position", [0.0, 0.0, 0.0]), dtype=np.float32).ravel()
        if position.shape[0] < 3:
            position = np.zeros(3, dtype=np.float32)
        velocity = np.asarray(getattr(msg, "velocity", [0.0, 0.0, 0.0]), dtype=np.float32).ravel()
        if velocity.shape[0] < 3:
            velocity = np.zeros(3, dtype=np.float32)

        quat_xyzw = self._last_lowstate_quat_xyzw.copy()
        try:
            imu = getattr(msg, "imu_state", None)
            if imu is not None and hasattr(imu, "quaternion"):
                q = np.asarray(imu.quaternion, dtype=np.float32).ravel()
                if q.shape[0] >= 4:
                    # Sport IMU quaternion follows Unitree convention wxyz.
                    cand = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
                    n = float(np.linalg.norm(cand))
                    if n > 1e-6:
                        quat_xyzw = (cand / n).astype(np.float32)
        except Exception:
            pass

        out = self._make_base_pose_msg(position[:3], velocity[:3], quat_xyzw)
        self.base_pose_publisher_.Write(out)

    def update_control_input(self):
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
        pass

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
    parser.add_argument("--keyboard", action="store_true", help="use keyboard DDS topic for state transitions and velocity commands")
    parser.add_argument("--keyboard-topic", type=str, default="rt/wireless_remote", help="DDS topic for keyboard command stream")
    parser.add_argument("--base-pose-topic", type=str, default="rt/base_pose", help="Unified base pose output topic for all consumers")
    args = parser.parse_args()

    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = Controller(
        config,
        use_mujoco=args.mujoco,
        keyboard=args.keyboard,
        keyboard_topic=args.keyboard_topic,
        base_pose_topic=args.base_pose_topic,
    )

    if not args.mujoco:
        controller.shut_down_control_service()

    # Run state machine
    try:
        while controller.state_machine.execute():
            pass
    finally:
        controller.close()

    print("Exit")
