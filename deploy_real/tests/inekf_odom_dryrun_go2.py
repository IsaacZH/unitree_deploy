import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEPLOY_REAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEPLOY_REAL_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_REAL_DIR)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowState_,
    unitree_go_msg_dds__SportModeState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as SportModeStateGo


@dataclass
class OdomEstimate:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_base: np.ndarray
    angular_velocity_base: np.ndarray


class InekfOdomEstimator:
    """Non-ROS reproduction of go2_odometry/scripts/inekf_odom.py core math."""

    def __init__(
        self,
        robot_freq: float = 500.0,
        base_frame: str = "base",
        contact_threshold: float = 20.0,
        gyroscope_noise: float = 0.01,
        accelerometer_noise: float = 0.1,
        gyroscope_bias_noise: float = 1.0e-5,
        accelerometer_bias_noise: float = 1.0e-4,
        contact_noise: float = 1.0e-3,
        joint_position_noise: float = 1.0e-3,
        contact_velocity_noise: float = 1.0e-3,
        reset_min_contact_count: int = 1,
        reset_loss_duration_s: float = 0.4,
        reinit_contact_duration_s: float = 0.15,
        urdf_path: Optional[str] = None,
    ):
        import pinocchio as pin
        from inekf import InEKF, Kinematics, NoiseParams, RobotState

        self.pin = pin
        self.InEKF = InEKF
        self.Kinematics = Kinematics

        self.dt = 1.0 / float(robot_freq)
        self.base_frame = base_frame
        self.contact_threshold = float(contact_threshold)
        self.pause = True
        self.reset_min_contact_count = int(reset_min_contact_count)

        # If contact quality is poor for long enough (e.g. sit/stand transition),
        # reset and wait for stable full contact before re-initializing.
        self._low_contact_steps = 0
        self._full_contact_steps = 0
        self._reset_loss_steps = max(1, int(float(reset_loss_duration_s) / self.dt))
        self._reinit_contact_steps = max(1, int(float(reinit_contact_duration_s) / self.dt))

        self.joint_pos_noise = float(joint_position_noise)
        self.contact_vel_noise = float(contact_velocity_noise)

        self.robot = self._load_robot(urdf_path)
        self.foot_frame_name = [prefix + "_foot" for prefix in ["FL", "FR", "RL", "RR"]]
        self.foot_frame_id = [self.robot.model.getFrameId(frame_name) for frame_name in self.foot_frame_name]
        self.imu_frame_id = self.robot.model.getFrameId("imu")
        self.base_frame_id = self.robot.model.getFrameId(self.base_frame)

        if self.imu_frame_id >= len(self.robot.model.frames):
            raise RuntimeError("Invalid imu frame id.")
        if self.base_frame_id >= len(self.robot.model.frames):
            raise RuntimeError(f"Invalid base frame id for '{self.base_frame}'.")

        self.pin.forwardKinematics(self.robot.model, self.robot.data, self.pin.neutral(self.robot.model))
        self.pin.updateFramePlacements(self.robot.model, self.robot.data)
        oMimu = self.robot.data.oMf[self.imu_frame_id]
        oMbase = self.robot.data.oMf[self.base_frame_id]
        self.imuMbase = oMimu.actInv(oMbase)

        gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        initial_state = RobotState()
        initial_state.setRotation(np.eye(3))
        initial_state.setVelocity(np.zeros(3))
        initial_state.setPosition(np.zeros(3))
        initial_state.setGyroscopeBias(np.zeros(3))
        initial_state.setAccelerometerBias(np.zeros(3))

        noise_params = NoiseParams()
        noise_params.setGyroscopeNoise(float(gyroscope_noise))
        noise_params.setAccelerometerNoise(float(accelerometer_noise))
        noise_params.setGyroscopeBiasNoise(float(gyroscope_bias_noise))
        noise_params.setAccelerometerBiasNoise(float(accelerometer_bias_noise))
        noise_params.setContactNoise(float(contact_noise))

        self.filter = InEKF(initial_state, noise_params)
        self.filter.setGravity(gravity)

    def _load_robot(self, urdf_path: Optional[str]):
        # Prefer the same loader as go2_odometry. Fallback to direct URDF path.
        try:
            from unitree_description.loader import loadGo2

            return loadGo2()
        except Exception:
            if urdf_path is None:
                raise RuntimeError(
                    "Failed to import unitree_description.loader.loadGo2 and no --urdf-path provided."
                )

            class _Robot:
                def __init__(self, model):
                    self.model = model
                    self.data = model.createData()

            model = self.pin.buildModelFromUrdf(urdf_path, self.pin.JointModelFreeFlyer())
            return _Robot(model)

    @staticmethod
    def _unitree_to_urdf_vec(vec):
        return [
            vec[3],
            vec[4],
            vec[5],
            vec[0],
            vec[1],
            vec[2],
            vec[9],
            vec[10],
            vec[11],
            vec[6],
            vec[7],
            vec[8],
        ]

    def _get_qvf_pinocchio(self, state_msg):
        q_unitree = [j.q for j in state_msg.motor_state[:12]]
        v_unitree = [j.dq for j in state_msg.motor_state[:12]]
        f_unitree = state_msg.foot_force

        q_pin = np.array([0.0] * 6 + [1.0] + self._unitree_to_urdf_vec(q_unitree), dtype=np.float64)
        v_pin = np.array([0.0] * 6 + self._unitree_to_urdf_vec(v_unitree), dtype=np.float64)
        f_pin = [float(f_unitree[i]) for i in [1, 0, 3, 2]]
        return q_pin, v_pin, f_pin

    def _initialize_filter(self, state_msg):
        self._low_contact_steps = 0
        self._full_contact_steps = 0
        q, v, _ = self._get_qvf_pinocchio(state_msg)

        # Unitree IMU quat is wxyz, Pinocchio free-flyer quaternion is xyzw.
        q[3] = float(state_msg.imu_state.quaternion[1])
        q[4] = float(state_msg.imu_state.quaternion[2])
        q[5] = float(state_msg.imu_state.quaternion[3])
        q[6] = float(state_msg.imu_state.quaternion[0])
        q[3:7] /= max(float(np.linalg.norm(q[3:7])), 1.0e-12)

        self.pin.forwardKinematics(self.robot.model, self.robot.data, q, v)
        self.pin.updateFramePlacements(self.robot.model, self.robot.data)

        oMbase = self.robot.data.oMf[self.base_frame_id]
        rpy = self.pin.rpy.matrixToRpy(oMbase.rotation)
        rpy[2] = 0.0
        oMbase.rotation = self.pin.rpy.rpyToMatrix(rpy)

        z_avg = 0.0
        for i in range(4):
            oMfoot = self.robot.data.oMf[self.foot_frame_id[i]]
            z_avg += float(oMfoot.translation[2])
        z_avg /= 4.0

        oMbase.translation[:2] = np.zeros(2)
        oMbase.translation[2] -= z_avg - 0.025

        oMimu = oMbase.act(self.imuMbase.inverse())

        state = self.filter.getState()
        state.setRotation(oMimu.rotation)
        state.setPosition(oMimu.translation)
        self.filter.setState(state)

    def _feet_transformations(self, state_msg):
        q_pin, v_pin, f_pin = self._get_qvf_pinocchio(state_msg)

        self.pin.forwardKinematics(self.robot.model, self.robot.data, q_pin, v_pin)
        self.pin.updateFramePlacements(self.robot.model, self.robot.data)
        self.pin.computeJointJacobians(self.robot.model, self.robot.data)

        oMimu = self.robot.data.oMf[self.imu_frame_id]
        contact_list = [bool(f >= self.contact_threshold) for f in f_pin]
        pose_list = []
        normed_covariance_list = []

        for i in range(4):
            oMfoot = self.robot.data.oMf[self.foot_frame_id[i]]
            imuMfoot = oMimu.actInv(oMfoot)
            pose_list.append(imuMfoot)

            Jc = self.pin.getFrameJacobian(
                self.robot.model,
                self.robot.data,
                self.foot_frame_id[i],
                self.pin.LOCAL,
            )[:3, 6:]
            normed_covariance_list.append(Jc @ Jc.transpose())

        return contact_list, pose_list, normed_covariance_list

    def _state_to_odom(self, filter_state, gyro_xyz):
        oMimu = self.pin.SE3(filter_state.getRotation(), filter_state.getPosition())
        v_linear_imu_world = filter_state.getX()[0:3, 3].reshape(-1)
        v_linear_imu_local = oMimu.inverse().rotation @ v_linear_imu_world
        v_imu_local = self.pin.Motion(linear=v_linear_imu_local, angular=gyro_xyz)

        base_pose = oMimu.act(self.imuMbase)
        base_velocity = self.imuMbase.actInv(v_imu_local)

        q_base = self.pin.Quaternion(base_pose.rotation)
        q_base.normalize()
        quat_wxyz = np.array([q_base.w, q_base.x, q_base.y, q_base.z], dtype=np.float64)

        return OdomEstimate(
            position=np.asarray(base_pose.translation, dtype=np.float64).reshape(3),
            quaternion_wxyz=quat_wxyz,
            linear_velocity_base=np.asarray(base_velocity.linear, dtype=np.float64).reshape(3),
            angular_velocity_base=np.asarray(base_velocity.angular, dtype=np.float64).reshape(3),
        )

    def update(self, lowstate_msg) -> Optional[OdomEstimate]:
        imu_state = np.concatenate(
            [
                np.asarray(lowstate_msg.imu_state.gyroscope, dtype=np.float64).reshape(3),
                np.asarray(lowstate_msg.imu_state.accelerometer, dtype=np.float64).reshape(3),
            ]
        )
        contact_list, pose_list, normed_covariance_list = self._feet_transformations(lowstate_msg)
        contact_count = int(sum(contact_list))
        full_contact = bool(all(contact_list))

        if self.pause:
            if full_contact:
                self._full_contact_steps += 1
                if self._full_contact_steps >= self._reinit_contact_steps:
                    self.pause = False
                    self._initialize_filter(lowstate_msg)
                    print("[InekfDryRun] Full contact stable, filter re-initialized.")
            else:
                self._full_contact_steps = 0
                return None

        if contact_count <= self.reset_min_contact_count:
            self._low_contact_steps += 1
        else:
            self._low_contact_steps = 0

        if self._low_contact_steps >= self._reset_loss_steps:
            if not self.pause:
                print(
                    "[InekfDryRun] Contact lost for too long "
                    f"(count={contact_count}), resetting filter and waiting for stable full contact."
                )
            self.pause = True
            self._full_contact_steps = 0
            return None

        self.filter.propagate(imu_state, self.dt)

        contact_pairs = []
        kinematics_list = []
        for i in range(4):
            contact_pairs.append((i, contact_list[i]))
            kinematics = self.Kinematics(
                i,
                pose_list[i].translation,
                self.joint_pos_noise * normed_covariance_list[i],
                np.zeros(3),
                self.contact_vel_noise * np.eye(3),
            )
            kinematics_list.append(kinematics)

        self.filter.setContacts(contact_pairs)
        self.filter.correctKinematics(kinematics_list)

        gyro = np.asarray(lowstate_msg.imu_state.gyroscope, dtype=np.float64).reshape(3)
        return self._state_to_odom(self.filter.getState(), gyro)


class InekfDryRunRunner:
    def __init__(self, args):
        self.args = args
        self.low_state = unitree_go_msg_dds__LowState_()
        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.last_odom: Optional[OdomEstimate] = None
        self._lock = threading.Lock()
        self._msg_count = 0

        self.foxglove_server = None
        self.foxglove_channels = {}

        self.estimator = InekfOdomEstimator(
            robot_freq=args.robot_freq,
            base_frame=args.base_frame,
            contact_threshold=args.contact_threshold,
            gyroscope_noise=args.gyroscope_noise,
            accelerometer_noise=args.accelerometer_noise,
            gyroscope_bias_noise=args.gyroscope_bias_noise,
            accelerometer_bias_noise=args.accelerometer_bias_noise,
            contact_noise=args.contact_noise,
            joint_position_noise=args.joint_position_noise,
            contact_velocity_noise=args.contact_velocity_noise,
            reset_min_contact_count=args.reset_min_contact_count,
            reset_loss_duration_s=args.reset_loss_duration,
            reinit_contact_duration_s=args.reinit_contact_duration,
            urdf_path=args.urdf_path,
        )

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)
        self.highstate_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeStateGo)
        self.highstate_subscriber.Init(self.high_state_handler, 10)

        self._init_foxglove()

    def low_state_handler(self, msg: LowStateGo):
        odom = self.estimator.update(msg)
        with self._lock:
            self.low_state = msg
            self._msg_count += 1
            if odom is not None:
                self.last_odom = odom

    def high_state_handler(self, msg: SportModeStateGo):
        with self._lock:
            self.high_state = msg

    @staticmethod
    def _extract_robot_pose_from_high_state(high_state):
        """Extract position and quaternion from SportModeState.
        Returns: (position_xyz, quaternion_wxyz)
        """
        position = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # Extract position
        if hasattr(high_state, "position"):
            pos = np.asarray(high_state.position, dtype=np.float64).ravel()
            if pos.shape[0] >= 3:
                position = pos[:3]
        
        # Extract quaternion from IMU state
        if hasattr(high_state, "imu_state") and hasattr(high_state.imu_state, "quaternion"):
            quat = np.asarray(high_state.imu_state.quaternion, dtype=np.float64).ravel()
            if quat.shape[0] >= 4:
                # Assuming stored as [x, y, z, w], convert to [w, x, y, z]
                quaternion = np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)
        
        return position, quaternion

    def _init_foxglove(self):
        if self.args.disable_foxglove:
            print("[InekfDryRun] Foxglove publishing disabled by flag.")
            return

        try:
            import foxglove
            import json as _json

            # JSON Schema for foxglove.FrameTransform
            # See: https://docs.foxglove.dev/docs/visualization/message-schemas/frame-transform
            _ft_schema = {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "object",
                        "properties": {
                            "sec": {"type": "integer"},
                            "nsec": {"type": "integer"},
                        },
                    },
                    "parent_frame_id": {"type": "string"},
                    "child_frame_id": {"type": "string"},
                    "translation": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                        },
                    },
                    "rotation": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                            "w": {"type": "number"},
                        },
                    },
                },
            }
            _ft_schema_obj = foxglove.Schema(
                name="foxglove.FrameTransform",
                encoding="jsonschema",
                data=_json.dumps(_ft_schema).encode(),
            )

            self.foxglove_server = foxglove.start_server(
                name="go2_odom_compare",
                host=self.args.foxglove_host,
                port=self.args.foxglove_port,
            )
            # Three separate topics, one per frame
            self.foxglove_channels = {
                "inekf": foxglove.Channel("/go2/tf/inekf", message_encoding="json", schema=_ft_schema_obj),
                "native": foxglove.Channel("/go2/tf/native", message_encoding="json", schema=_ft_schema_obj),
            }
            print(
                "[InekfDryRun] Foxglove server started at "
                f"ws://{self.args.foxglove_host}:{self.foxglove_server.port}"
            )
            print(
                "[InekfDryRun] Publishing foxglove.FrameTransform on:"
                " /go2/tf/inekf  /go2/tf/native"
            )
        except Exception as exc:
            self.foxglove_server = None
            self.foxglove_channels = {}
            print(f"[InekfDryRun] Foxglove init failed: {exc}")

    def _publish_foxglove(self, t_sec: float, inekf_pos: np.ndarray, inekf_quat: np.ndarray,
                           robot_pos: np.ndarray, robot_quat: np.ndarray):
        if self.foxglove_server is None:
            return

        ch_inekf = self.foxglove_channels.get("inekf")
        ch_native = self.foxglove_channels.get("native")
        if ch_inekf is None:
            return

        # Timestamp in sec/nsec for foxglove.FrameTransform
        timestamp_ns = int(t_sec * 1e9)
        stamp = {"sec": int(timestamp_ns // 1_000_000_000), "nsec": int(timestamp_ns % 1_000_000_000)}

        # /go2/tf/inekf — INEKF estimated frame
        ch_inekf.log({
            "timestamp": stamp,
            "parent_frame_id": "world",
            "child_frame_id": "go2_inekf",
            "translation": {
                "x": float(inekf_pos[0]),
                "y": float(inekf_pos[1]),
                "z": 0.3,
            },
            "rotation": {
                "x": float(inekf_quat[1]),  # wxyz -> xyzw
                "y": float(inekf_quat[2]),
                "z": float(inekf_quat[3]),
                "w": float(inekf_quat[0]),
            },
        })

        # /go2/tf/native — SportModeState native frame
        if ch_native is not None and np.all(np.isfinite(robot_pos)):
            ch_native.log({
                "timestamp": stamp,
                "parent_frame_id": "world",
                "child_frame_id": "go2_native",
                "translation": {
                    "x": float(robot_pos[0]),  
                    "y": float(robot_pos[1]),
                    "z": float(robot_pos[2]),
                },
                "rotation": {
                    "x": float(robot_quat[1]),  # wxyz -> xyzw
                    "y": float(robot_quat[2]),
                    "z": float(robot_quat[3]),
                    "w": float(robot_quat[0]),
                },
            })

    def close(self):
        for ch in self.foxglove_channels.values():
            try:
                ch.close()
            except Exception:
                pass

        if self.foxglove_server is not None:
            try:
                self.foxglove_server.stop()
            except Exception:
                pass

    def wait_for_low_state(self, timeout_s: float = 5.0):
        start = time.monotonic()
        while self.low_state.tick == 0:
            if time.monotonic() - start > timeout_s:
                print("[InekfDryRun] LowState wait timeout, continue without blocking.")
                return
            time.sleep(0.002)
        print("[InekfDryRun] LowState connected.")

    def run(self):
        self.wait_for_low_state()
        print_period = 1.0 / max(self.args.print_hz, 1.0e-6)
        next_print = time.monotonic()

        while True:
            now = time.monotonic()
            if now >= next_print:
                with self._lock:
                    odom = self.last_odom
                    msg_count = self._msg_count
                    robot_pos, robot_quat = self._extract_robot_pose_from_high_state(self.high_state)
                if odom is None:
                    print(f"[InekfDryRun] waiting_filter_start msgs={msg_count}")
                else:
                    p = odom.position
                    q = odom.quaternion_wxyz
                    v = odom.linear_velocity_base
                    w = odom.angular_velocity_base
                    t_now = time.time()
                    if np.all(np.isfinite(robot_pos)):
                        pos_err = p - robot_pos
                        pos_err_norm = float(np.linalg.norm(pos_err))
                        pos_cmp_text = (
                            f" robot_pos=({robot_pos[1]: .3f},{robot_pos[0]: .3f},{robot_pos[2]: .3f})"
                            f" pos_err=({pos_err[1]: .3f},{pos_err[0]: .3f},{pos_err[2]: .3f})"
                            f" |err|={pos_err_norm: .3f}"
                        )
                    else:
                        pos_cmp_text = " robot_pos=( nan, nan, nan) pos_err=( nan, nan, nan) |err|= nan"

                    self._publish_foxglove(t_now, p, q, robot_pos, robot_quat)

                    # print(
                    #     "[InekfDryRun] "
                    #     f"t={t_now:.3f} msgs={msg_count} "
                    #     f"pos=({p[1]: .3f},{p[0]: .3f},{p[2]: .3f}) "
                    #     f"quat_wxyz=({q[0]: .4f},{q[1]: .4f},{q[2]: .4f},{q[3]: .4f}) "
                    #     f"v_b=({v[0]: .3f},{v[1]: .3f},{v[2]: .3f}) "
                    #     f"w_b=({w[0]: .3f},{w[1]: .3f},{w[2]: .3f})"
                    #     f"{pos_cmp_text}"
                    # )
                next_print = now + print_period

            time.sleep(0.001)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Non-ROS INEKF odometry dry-run for Unitree Go2 from rt/lowstate DDS stream."
    )
    parser.add_argument("net", type=str, help="network interface, e.g. eno1")
    parser.add_argument("--print-hz", type=float, default=10.0, help="print frequency")
    parser.add_argument("--robot-freq", type=float, default=500.0, help="expected lowstate rate (Hz)")
    parser.add_argument("--base-frame", type=str, default="base", help="base frame name in URDF")
    parser.add_argument("--contact-threshold", type=float, default=20.0, help="foot contact threshold")

    parser.add_argument("--gyroscope-noise", type=float, default=0.01)
    parser.add_argument("--accelerometer-noise", type=float, default=0.1)
    parser.add_argument("--gyroscope-bias-noise", type=float, default=1.0e-5)
    parser.add_argument("--accelerometer-bias-noise", type=float, default=1.0e-4)
    parser.add_argument("--contact-noise", type=float, default=1.0e-3)
    parser.add_argument("--joint-position-noise", type=float, default=1.0e-3)
    parser.add_argument("--contact-velocity-noise", type=float, default=1.0e-3)
    parser.add_argument(
        "--reset-min-contact-count",
        type=int,
        default=1,
        help="reset filter when contact count stays <= this value for reset-loss-duration",
    )
    parser.add_argument(
        "--reset-loss-duration",
        type=float,
        default=0.4,
        help="seconds of low contact required to trigger filter reset",
    )
    parser.add_argument(
        "--reinit-contact-duration",
        type=float,
        default=0.15,
        help="seconds of full contact required before re-initializing filter",
    )

    parser.add_argument(
        "--urdf-path",
        type=str,
        default="/home/isaac/deploy_him_py/go2_description/urdf/go2_description.urdf",
        help="Fallback URDF path used when unitree_description.loader.loadGo2 is unavailable.",
    )
    parser.add_argument("--foxglove-host", type=str, default="127.0.0.1", help="Foxglove websocket host")
    parser.add_argument("--foxglove-port", type=int, default=8765, help="Foxglove websocket port (0 for auto)")
    parser.add_argument("--disable-foxglove", action="store_true", help="disable Foxglove publishing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ChannelFactoryInitialize(0, args.net)

    runner = InekfDryRunRunner(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[InekfDryRun] Exit.")
    finally:
        runner.close()