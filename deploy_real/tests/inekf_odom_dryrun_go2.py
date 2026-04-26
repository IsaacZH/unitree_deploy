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
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo


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

        if self.pause:
            if all(contact_list):
                self.pause = False
                self._initialize_filter(lowstate_msg)
                print("[InekfDryRun] All feet in contact, starting filter.")
            else:
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
        self.last_odom: Optional[OdomEstimate] = None
        self._lock = threading.Lock()
        self._msg_count = 0

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
            urdf_path=args.urdf_path,
        )

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowStateGo):
        odom = self.estimator.update(msg)
        with self._lock:
            self.low_state = msg
            self._msg_count += 1
            if odom is not None:
                self.last_odom = odom

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
                if odom is None:
                    print(f"[InekfDryRun] waiting_filter_start msgs={msg_count}")
                else:
                    p = odom.position
                    q = odom.quaternion_wxyz
                    v = odom.linear_velocity_base
                    w = odom.angular_velocity_base
                    print(
                        "[InekfDryRun] "
                        f"t={time.time():.3f} msgs={msg_count} "
                        f"pos=({p[0]: .3f},{p[1]: .3f},{p[2]: .3f}) "
                        f"quat_wxyz=({q[0]: .4f},{q[1]: .4f},{q[2]: .4f},{q[3]: .4f}) "
                        f"v_b=({v[0]: .3f},{v[1]: .3f},{v[2]: .3f}) "
                        f"w_b=({w[0]: .3f},{w[1]: .3f},{w[2]: .3f})"
                    )
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
        "--urdf-path",
        type=str,
        default=None,
        help="Optional fallback URDF path if unitree_description.loader.loadGo2 is unavailable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ChannelFactoryInitialize(0, args.net)

    runner = InekfDryRunRunner(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[InekfDryRun] Exit.")