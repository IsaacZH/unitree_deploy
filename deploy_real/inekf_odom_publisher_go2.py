import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEPLOY_REAL_DIR = os.path.dirname(os.path.abspath(__file__))
if DEPLOY_REAL_DIR not in sys.path:
	sys.path.insert(0, DEPLOY_REAL_DIR)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
	nav_msgs_msg_dds__Odometry_,
	unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as OdometryGeo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo


@dataclass
class OdomEstimate:
	position: np.ndarray
	quaternion_wxyz: np.ndarray
	linear_velocity_base: np.ndarray
	angular_velocity_base: np.ndarray


class InekfOdomEstimator:
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
		self.Kinematics = Kinematics

		self.dt = 1.0 / float(robot_freq)
		self.base_frame = base_frame
		self.contact_threshold = float(contact_threshold)
		self.pause = True
		self.reset_min_contact_count = int(reset_min_contact_count)

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
					print("[InekfOdomPub] Full contact stable, filter initialized.")
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
					"[InekfOdomPub] Contact lost for too long "
					f"(count={contact_count}), resetting filter."
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


class InekfOdomPublisher:
	def __init__(self, args):
		self.args = args
		self.low_state = unitree_go_msg_dds__LowState_()
		self.msg_count = 0
		self.pub_count = 0
		self.last_print_t = time.monotonic()
		self._pub_ready = False
		self.foxglove_server = None
		self.foxglove_channel = None

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

		self.pub = ChannelPublisher(args.odom_topic, OdometryGeo)
		self.pub.Init()
		self._pub_ready = True
		self._init_foxglove()

		self.sub = ChannelSubscriber("rt/lowstate", LowStateGo)
		self.sub.Init(self.low_state_handler, 10)

	def _init_foxglove(self):
		if not self.args.publish_foxglove:
			return

		try:
			import foxglove
			import json as _json

			ft_schema = {
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

			ft_schema_obj = foxglove.Schema(
				name="foxglove.FrameTransform",
				encoding="jsonschema",
				data=_json.dumps(ft_schema).encode(),
			)

			self.foxglove_server = foxglove.start_server(
				name="go2_inekf_odom_publisher",
				host=self.args.foxglove_host,
				port=self.args.foxglove_port,
			)
			self.foxglove_channel = foxglove.Channel(
				self.args.foxglove_topic,
				message_encoding="json",
				schema=ft_schema_obj,
			)
			print(
				"[InekfOdomPub] Foxglove enabled at "
				f"ws://{self.args.foxglove_host}:{self.foxglove_server.port} "
				f"topic={self.args.foxglove_topic}"
			)
		except Exception as exc:
			self.foxglove_server = None
			self.foxglove_channel = None
			print(f"[InekfOdomPub] Foxglove init failed: {exc}")

	def _publish_foxglove(self, est: OdomEstimate, t_sec: float):
		if self.foxglove_channel is None:
			return

		timestamp_ns = int(t_sec * 1e9)
		stamp = {"sec": int(timestamp_ns // 1_000_000_000), "nsec": int(timestamp_ns % 1_000_000_000)}

		try:
			self.foxglove_channel.log({
				"timestamp": stamp,
				"parent_frame_id": self.args.world_frame,
				"child_frame_id": self.args.base_frame_out,
				"translation": {
					"x": float(est.position[0]),
					"y": float(est.position[1]),
					"z": float(est.position[2]),
				},
				"rotation": {
					"x": float(est.quaternion_wxyz[1]),
					"y": float(est.quaternion_wxyz[2]),
					"z": float(est.quaternion_wxyz[3]),
					"w": float(est.quaternion_wxyz[0]),
				},
			})
		except Exception as exc:
			print(f"[InekfOdomPub] Foxglove publish failed: {exc}")

	def _build_odom_msg(self, est: OdomEstimate, t_sec: float):
		msg = nav_msgs_msg_dds__Odometry_()

		stamp_ns = int(t_sec * 1e9)
		msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
		msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
		msg.header.frame_id = self.args.world_frame
		msg.child_frame_id = self.args.base_frame_out

		msg.pose.pose.position.x = float(est.position[0])
		msg.pose.pose.position.y = float(est.position[1])
		msg.pose.pose.position.z = float(est.position[2])

		msg.pose.pose.orientation.w = float(est.quaternion_wxyz[0])
		msg.pose.pose.orientation.x = float(est.quaternion_wxyz[1])
		msg.pose.pose.orientation.y = float(est.quaternion_wxyz[2])
		msg.pose.pose.orientation.z = float(est.quaternion_wxyz[3])

		msg.twist.twist.linear.x = float(est.linear_velocity_base[0])
		msg.twist.twist.linear.y = float(est.linear_velocity_base[1])
		msg.twist.twist.linear.z = float(est.linear_velocity_base[2])

		msg.twist.twist.angular.x = float(est.angular_velocity_base[0])
		msg.twist.twist.angular.y = float(est.angular_velocity_base[1])
		msg.twist.twist.angular.z = float(est.angular_velocity_base[2])

		return msg

	def low_state_handler(self, msg: LowStateGo):
		self.low_state = msg
		self.msg_count += 1

		est = self.estimator.update(msg)
		if est is None:
			return
		if not self._pub_ready:
			return

		now_wall = time.time()
		odom_msg = self._build_odom_msg(est, now_wall)
		try:
			self.pub.Write(odom_msg)
		except Exception as exc:
			print(f"[InekfOdomPub] publish failed: {exc}")
			return
		self.pub_count += 1
		self._publish_foxglove(est, now_wall)

		now = time.monotonic()
		if now - self.last_print_t >= 1.0 / max(self.args.print_hz, 1.0e-6):
			self.last_print_t = now
			p = est.position
			print(
				"[InekfOdomPub] "
				f"in={self.msg_count} out={self.pub_count} topic={self.args.odom_topic} "
				f"pos=({p[0]: .3f},{p[1]: .3f},{p[2]: .3f})"
			)

	def wait_for_low_state(self, timeout_s: float = 5.0):
		start = time.monotonic()
		while self.low_state.tick == 0:
			if time.monotonic() - start > timeout_s:
				print("[InekfOdomPub] LowState wait timeout, continue running.")
				return
			time.sleep(0.002)
		print("[InekfOdomPub] LowState connected.")

	def run(self):
		self.wait_for_low_state()
		while True:
			time.sleep(0.2)

	def close(self):
		if self.foxglove_channel is not None:
			try:
				self.foxglove_channel.close()
			except Exception:
				pass

		if self.foxglove_server is not None:
			try:
				self.foxglove_server.stop()
			except Exception:
				pass


def parse_args():
	parser = argparse.ArgumentParser(
		description="Production INEKF odometry publisher: subscribe rt/lowstate and publish Odometry DDS."
	)
	parser.add_argument("net", type=str, help="network interface, e.g. eno1")
	parser.add_argument("--odom-topic", type=str, default="rt/inekf/odom", help="output odom DDS topic")
	parser.add_argument("--world-frame", type=str, default="world", help="Odometry header frame_id")
	parser.add_argument("--base-frame-out", type=str, default="base", help="Odometry child_frame_id")
	parser.add_argument("--print-hz", type=float, default=5.0, help="status print frequency")
	parser.add_argument("--publish-foxglove", action="store_true", help="also publish FrameTransform to Foxglove")
	parser.add_argument("--foxglove-host", type=str, default="127.0.0.1", help="Foxglove websocket host")
	parser.add_argument("--foxglove-port", type=int, default=44911, help="Foxglove websocket port (0 for auto)")
	parser.add_argument("--foxglove-topic", type=str, default="/go2/tf/inekf", help="Foxglove topic for transform")

	parser.add_argument("--robot-freq", type=float, default=500.0)
	parser.add_argument("--base-frame", type=str, default="base")
	parser.add_argument("--contact-threshold", type=float, default=20.0)

	parser.add_argument("--gyroscope-noise", type=float, default=0.01)
	parser.add_argument("--accelerometer-noise", type=float, default=0.1)
	parser.add_argument("--gyroscope-bias-noise", type=float, default=1.0e-5)
	parser.add_argument("--accelerometer-bias-noise", type=float, default=1.0e-4)
	parser.add_argument("--contact-noise", type=float, default=1.0e-3)
	parser.add_argument("--joint-position-noise", type=float, default=1.0e-3)
	parser.add_argument("--contact-velocity-noise", type=float, default=1.0e-3)

	parser.add_argument("--reset-min-contact-count", type=int, default=1)
	parser.add_argument("--reset-loss-duration", type=float, default=0.4)
	parser.add_argument("--reinit-contact-duration", type=float, default=0.15)

	parser.add_argument(
		"--urdf-path",
		type=str,
		default="/home/isaac/deploy_him_py/go2_description/urdf/go2_description.urdf",
		help="Fallback URDF path used when unitree_description.loader.loadGo2 is unavailable.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	ChannelFactoryInitialize(0, args.net)

	runner = InekfOdomPublisher(args)
	try:
		runner.run()
	except KeyboardInterrupt:
		print("\n[InekfOdomPub] Exit.")
	finally:
		runner.close()
