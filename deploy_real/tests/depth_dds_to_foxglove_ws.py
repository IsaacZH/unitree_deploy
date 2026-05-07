import argparse
import base64
import fcntl
import json
import math
import os
import socket
import struct
import sys
import threading
import time

import numpy as np
from matplotlib import cm

DEPLOY_REAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEPLOY_REAL_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_REAL_DIR)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from common.depth_image_sub import DepthImage_, _decode_depth_message
from common.nav_debug_dds import NavDebug_
from common.nav_target_dds import NavTarget_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as OdometryGeo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as SportModeStateGo


def _resolve_ipv4_from_interface(interface_name: str) -> str:
    """Resolve IPv4 address for a Linux network interface."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        ifname = interface_name[:15].encode("utf-8")
        req = struct.pack("256s", ifname)
        res = fcntl.ioctl(sock.fileno(), 0x8915, req)  # SIOCGIFADDR
        return socket.inet_ntoa(res[20:24])
    finally:
        sock.close()


class DepthBridge:
    def __init__(self, args):
        self.args = args
        self._lock = threading.Lock()
        self._latest_depth = None
        self._latest_scale = None
        self._latest_stamp = None
        self._latest_depth_width = 0
        self._latest_depth_height = 0
        self._latest_depth_fx = 0.0
        self._latest_depth_fy = 0.0
        self._latest_depth_cx = 0.0
        self._latest_depth_cy = 0.0
        self._frames = 0
        self._latest_robot_pos = np.zeros(3, dtype=np.float32)
        self._latest_robot_yaw = 0.0
        self._latest_robot_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._latest_target_pos = np.zeros(3, dtype=np.float32)
        self._latest_cmd_body = np.zeros(3, dtype=np.float32)
        self._plasma_cmap = cm.get_cmap("plasma")
        self._q_robot_to_cam = self._robot_to_cam_quat_xyzw()

        self._init_foxglove()
        self._init_dds()

    def _init_foxglove(self):
        import foxglove

        host = self.args.foxglove_host
        if self.args.foxglove_interface:
            host = _resolve_ipv4_from_interface(self.args.foxglove_interface)
        self.foxglove_host = host

        compressed_schema = {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "object",
                    "properties": {
                        "sec": {"type": "integer"},
                        "nsec": {"type": "integer"},
                    },
                },
                "frame_id": {"type": "string"},
                "format": {"type": "string"},
                "data": {"type": "string", "contentEncoding": "base64"},
            },
        }

        frame_transform_schema = {
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

        pose_in_frame_schema = {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "object",
                    "properties": {
                        "sec": {"type": "integer"},
                        "nsec": {"type": "integer"},
                    },
                },
                "frame_id": {"type": "string"},
                "pose": {
                    "type": "object",
                    "properties": {
                        "position": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                            },
                        },
                        "orientation": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "w": {"type": "number"},
                            },
                        },
                    },
                },
            },
        }

        scene_update_schema = {
            "type": "object",
            "properties": {
                "deletions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "timestamp": {
                                "type": "object",
                                "properties": {
                                    "sec": {"type": "integer"},
                                    "nsec": {"type": "integer"},
                                },
                            },
                            "frame_id": {"type": "string"},
                            "arrows": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "pose": {
                                            "type": "object",
                                            "properties": {
                                                "position": {
                                                    "type": "object",
                                                    "properties": {
                                                        "x": {"type": "number"},
                                                        "y": {"type": "number"},
                                                        "z": {"type": "number"},
                                                    },
                                                },
                                                "orientation": {
                                                    "type": "object",
                                                    "properties": {
                                                        "x": {"type": "number"},
                                                        "y": {"type": "number"},
                                                        "z": {"type": "number"},
                                                        "w": {"type": "number"},
                                                    },
                                                },
                                            },
                                        },
                                        "shaft_length": {"type": "number"},
                                        "shaft_diameter": {"type": "number"},
                                        "head_length": {"type": "number"},
                                        "head_diameter": {"type": "number"},
                                        "color": {
                                            "type": "object",
                                            "properties": {
                                                "r": {"type": "number"},
                                                "g": {"type": "number"},
                                                "b": {"type": "number"},
                                                "a": {"type": "number"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

        pointcloud_schema = {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "object",
                    "properties": {
                        "sec": {"type": "integer"},
                        "nsec": {"type": "integer"},
                    },
                },
                "frame_id": {"type": "string"},
                "pose": {
                    "type": "object",
                    "properties": {
                        "position": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                            },
                        },
                        "orientation": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "w": {"type": "number"},
                            },
                        },
                    },
                },
                "point_stride": {"type": "integer"},
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "offset": {"type": "integer"},
                            "type": {"type": "integer"},
                        },
                    },
                },
                "data": {"type": "string", "contentEncoding": "base64"},
            },
        }

        self.foxglove_server = foxglove.start_server(
            name="go2_depth_bridge",
            host=self.foxglove_host,
            port=self.args.foxglove_port,
        )

        compressed_schema_obj = foxglove.Schema(
            name="foxglove.CompressedImage",
            encoding="jsonschema",
            data=json.dumps(compressed_schema).encode(),
        )
        tf_schema_obj = foxglove.Schema(
            name="foxglove.FrameTransform",
            encoding="jsonschema",
            data=json.dumps(frame_transform_schema).encode(),
        )
        pose_schema_obj = foxglove.Schema(
            name="foxglove.PoseInFrame",
            encoding="jsonschema",
            data=json.dumps(pose_in_frame_schema).encode(),
        )
        scene_schema_obj = foxglove.Schema(
            name="foxglove.SceneUpdate",
            encoding="jsonschema",
            data=json.dumps(scene_update_schema).encode(),
        )
        pointcloud_schema_obj = foxglove.Schema(
            name="foxglove.PointCloud",
            encoding="jsonschema",
            data=json.dumps(pointcloud_schema).encode(),
        )
        self.ch_image = foxglove.Channel(
            self.args.image_channel,
            message_encoding="json",
            schema=compressed_schema_obj,
        )
        self.ch_tf_base = foxglove.Channel(
            self.args.tf_base_channel,
            message_encoding="json",
            schema=tf_schema_obj,
        )
        self.ch_tf_target = foxglove.Channel(
            self.args.tf_target_channel,
            message_encoding="json",
            schema=tf_schema_obj,
        )
        self.ch_cmd_pose = None
        if self.args.enable_cmd_pose:
            self.ch_cmd_pose = foxglove.Channel(
                self.args.cmd_pose_topic,
                message_encoding="json",
                schema=pose_schema_obj,
            )
        self.ch_cmd_arrow = foxglove.Channel(
            self.args.cmd_arrow_topic,
            message_encoding="json",
            schema=scene_schema_obj,
        )
        self.ch_pointcloud = None
        if not self.args.no_pointcloud:
            self.ch_pointcloud = foxglove.Channel(
                self.args.pointcloud_topic,
                message_encoding="json",
                schema=pointcloud_schema_obj,
            )

        print(
            "[DepthBridge] Foxglove WS started at "
            f"ws://{self.foxglove_host}:{self.foxglove_server.port}"
        )
        print(f"[DepthBridge] Image channel: {self.args.image_channel}")
        print(
            "[DepthBridge] TF channels: "
            f"{self.args.tf_base_channel}  {self.args.tf_target_channel}"
        )
        if self.args.enable_cmd_pose:
            print(f"[DepthBridge] Cmd dir topic: {self.args.cmd_pose_topic} (foxglove.PoseInFrame)")
        else:
            print("[DepthBridge] Cmd dir pose topic disabled")
        print(f"[DepthBridge] Cmd arrow topic: {self.args.cmd_arrow_topic} (foxglove.SceneUpdate)")
        if self.ch_pointcloud is not None:
            print(
                "[DepthBridge] Point cloud topic: "
                f"{self.args.pointcloud_topic} (foxglove.PointCloud) "
                f"downsample={self.args.pointcloud_downsample} max_points={self.args.pointcloud_max_points}"
            )
        else:
            print("[DepthBridge] Point cloud topic disabled")
        print(f"[DepthBridge] pose_source={self.args.pose_source}")

    def _init_dds(self):
        ChannelFactoryInitialize(0, self.args.net)
        self.depth_sub = ChannelSubscriber(self.args.depth_topic, DepthImage_)
        self.depth_sub.Init(self._depth_handler, 10)
        if self.args.pose_source == "inekf":
            self.odom_sub = ChannelSubscriber(self.args.inekf_odom_topic, OdometryGeo)
            self.odom_sub.Init(self._odom_handler, 10)
            print(
                "[DepthBridge] pose_source=inekf, subscribed "
                f"odom={self.args.inekf_odom_topic}"
            )
        else:
            self.lowstate_sub = ChannelSubscriber(self.args.lowstate_topic, LowStateGo)
            self.lowstate_sub.Init(self._lowstate_handler, 10)
            self.sport_sub = ChannelSubscriber(self.args.sport_topic, SportModeStateGo)
            self.sport_sub.Init(self._sport_handler, 10)
            print(
                "[DepthBridge] pose_source=sport, subscribed "
                f"lowstate={self.args.lowstate_topic} sport={self.args.sport_topic}"
            )
        self.nav_target_sub = ChannelSubscriber(self.args.nav_target_topic, NavTarget_)
        self.nav_target_sub.Init(self._nav_target_handler, 10)
        self.nav_debug_sub = ChannelSubscriber(self.args.nav_debug_topic, NavDebug_)
        self.nav_debug_sub.Init(self._nav_debug_handler, 10)
        print(f"[DepthBridge] Subscribed DDS topic: {self.args.depth_topic}")
        print(f"[DepthBridge] Subscribed DDS topic: {self.args.nav_target_topic}")
        print(f"[DepthBridge] Subscribed DDS topic: {self.args.nav_debug_topic}")

    def _depth_handler(self, msg: DepthImage_):
        try:
            depth = _decode_depth_message(msg).copy()
            with self._lock:
                self._latest_depth = depth
                self._latest_scale = float(msg.depth_scale)
                self._latest_stamp = time.time()
                self._latest_depth_width = int(msg.width)
                self._latest_depth_height = int(msg.height)
                self._latest_depth_fx = float(msg.intrinsics.fx)
                self._latest_depth_fy = float(msg.intrinsics.fy)
                self._latest_depth_cx = float(msg.intrinsics.cx)
                self._latest_depth_cy = float(msg.intrinsics.cy)
                self._frames += 1
        except Exception as exc:
            print(f"[DepthBridge] Decode error: {exc}")

    @staticmethod
    def _quat_multiply_xyzw(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = [float(v) for v in q1_xyzw[:4]]
        x2, y2, z2, w2 = [float(v) for v in q2_xyzw[:4]]
        out = np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float32,
        )
        n = float(np.linalg.norm(out))
        if n <= 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return (out / n).astype(np.float32)

    @staticmethod
    def _quat_xyzw_from_axis_angle(axis_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = np.asarray(axis_xyz, dtype=np.float32).ravel()[:3]
        n = float(np.linalg.norm(axis))
        if n <= 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        axis = axis / n
        s = math.sin(0.5 * angle_rad)
        c = math.cos(0.5 * angle_rad)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)

    @classmethod
    def _robot_to_cam_quat_xyzw(cls) -> np.ndarray:
        q_forward = cls._quat_xyzw_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.5 * math.pi)
        q_ccw_90 = cls._quat_xyzw_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), -0.5 * math.pi)
        return cls._quat_multiply_xyzw(q_forward, q_ccw_90)

    @staticmethod
    def _quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = [float(v) for v in quat_xyzw[:4]]
        n = x * x + y * y + z * z + w * w
        if n <= 1e-12:
            return np.eye(3, dtype=np.float32)
        s = 2.0 / n
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        return np.array(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dtype=np.float32,
        )

    def _depth_to_world_points(
        self,
        depth_u16: np.ndarray,
        depth_scale: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        robot_pos: np.ndarray,
        robot_quat_xyzw: np.ndarray,
    ) -> np.ndarray:
        step = max(1, int(self.args.pointcloud_downsample))
        depth_sub = depth_u16[::step, ::step].astype(np.float32) * float(depth_scale)
        if depth_sub.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        rows, cols = depth_sub.shape
        us = np.arange(0, cols * step, step, dtype=np.float32)
        vs = np.arange(0, rows * step, step, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)

        valid = (depth_sub > float(self.args.pointcloud_min_depth)) & (depth_sub < float(self.args.pointcloud_max_depth))
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        z = depth_sub[valid]
        x = (uu[valid] - float(cx)) / max(float(fx), 1e-6) * z
        y = (vv[valid] - float(cy)) / max(float(fy), 1e-6) * z
        points_cam = np.stack([x, y, z], axis=1).astype(np.float32)

        max_points = max(1, int(self.args.pointcloud_max_points))
        if points_cam.shape[0] > max_points:
            stride = int(math.ceil(points_cam.shape[0] / float(max_points)))
            points_cam = points_cam[::max(stride, 1)]

        q_world_cam = self._quat_multiply_xyzw(robot_quat_xyzw, self._q_robot_to_cam)
        rot_world_cam = self._quat_xyzw_to_rotmat(q_world_cam)
        return (points_cam @ rot_world_cam.T + robot_pos.reshape(1, 3)).astype(np.float32)

    def _update_orientation_from_wxyz_quat(self, quat: np.ndarray):
        if quat.shape[0] < 4:
            return
        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float32)
        norm = float(np.linalg.norm(quat_xyzw))
        if norm <= 1e-8:
            return

        quat_xyzw = (quat_xyzw / norm).astype(np.float32)
        self._latest_robot_quat_xyzw = quat_xyzw
        self._latest_robot_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    def _lowstate_handler(self, msg: LowStateGo):
        try:
            if hasattr(msg, "imu_state") and hasattr(msg.imu_state, "quaternion"):
                quat = np.asarray(msg.imu_state.quaternion, dtype=np.float32).ravel()
                with self._lock:
                    self._update_orientation_from_wxyz_quat(quat)
        except Exception as exc:
            print(f"[DepthBridge] LowState decode error: {exc}")

    def _sport_handler(self, msg: SportModeStateGo):
        try:
            with self._lock:
                if hasattr(msg, "position"):
                    pos = np.asarray(msg.position, dtype=np.float32).ravel()
                    if pos.shape[0] >= 3:
                        self._latest_robot_pos = pos[:3].copy()

                if hasattr(msg, "imu_state") and hasattr(msg.imu_state, "quaternion"):
                    quat = np.asarray(msg.imu_state.quaternion, dtype=np.float32).ravel()
                    # Some simulators publish zeros here; keep this as fallback.
                    self._update_orientation_from_wxyz_quat(quat)
        except Exception as exc:
            print(f"[DepthBridge] Sport decode error: {exc}")

    def _odom_handler(self, msg: OdometryGeo):
        try:
            pos = np.array(
                [
                    float(msg.pose.pose.position.x),
                    float(msg.pose.pose.position.y),
                    float(msg.pose.pose.position.z),
                ],
                dtype=np.float32,
            )
            qx = float(msg.pose.pose.orientation.x)
            qy = float(msg.pose.pose.orientation.y)
            qz = float(msg.pose.pose.orientation.z)
            qw = float(msg.pose.pose.orientation.w)
            quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float32)
            norm = float(np.linalg.norm(quat_xyzw))
            if norm <= 1e-8:
                return

            quat_xyzw = (quat_xyzw / norm).astype(np.float32)
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
            with self._lock:
                self._latest_robot_pos = pos
                self._update_orientation_from_wxyz_quat(quat_wxyz)
        except Exception as exc:
            print(f"[DepthBridge] Odom decode error: {exc}")

    def _nav_target_handler(self, msg: NavTarget_):
        try:
            target = np.asarray(msg.target_world, dtype=np.float32).ravel()
            if target.shape[0] >= 3:
                with self._lock:
                    self._latest_target_pos = target[:3].copy()
        except Exception as exc:
            print(f"[DepthBridge] Nav target decode error: {exc}")

    def _nav_debug_handler(self, msg: NavDebug_):
        try:
            cmd = np.asarray(msg.target_speed_b, dtype=np.float32).ravel()
            if cmd.shape[0] >= 3:
                with self._lock:
                    self._latest_cmd_body = cmd[:3].copy()
        except Exception as exc:
            print(f"[DepthBridge] Nav debug decode error: {exc}")

    @staticmethod
    def _to_stamp(ts: float):
        ns = int(ts * 1e9)
        return {"sec": int(ns // 1_000_000_000), "nsec": int(ns % 1_000_000_000)}

    @staticmethod
    def _yaw_to_quat_xyzw(yaw: float):
        return [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]

    @staticmethod
    def _tf_msg(stamp, child_frame_id: str, pos_xyz: np.ndarray, quat_xyzw):
        return {
            "timestamp": stamp,
            "parent_frame_id": "world",
            "child_frame_id": child_frame_id,
            "translation": {
                "x": float(pos_xyz[0]),
                "y": float(pos_xyz[1]),
                "z": float(pos_xyz[2]),
            },
            "rotation": {
                "x": float(quat_xyzw[0]),
                "y": float(quat_xyzw[1]),
                "z": float(quat_xyzw[2]),
                "w": float(quat_xyzw[3]),
            },
        }

    def _encode_depth_png(self, depth_u16: np.ndarray, scale: float) -> bytes:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("opencv-python is required for PNG encoding") from exc

        depth_m = depth_u16.astype(np.float32) * scale
        valid = depth_m > 0.0
        if np.any(valid):
            low = self.args.visualize_min_depth
            high = self.args.visualize_max_depth
            depth_clip = np.clip(depth_m, low, high)
            norm = (depth_clip - low) / max(high - low, 1e-6)
            rgba = self._plasma_cmap(norm)  # [H, W, 4], values in [0, 1]
            rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
            vis = rgb[..., ::-1].copy()  # RGB -> BGR for OpenCV
            vis[~valid] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            vis = np.zeros((depth_u16.shape[0], depth_u16.shape[1], 3), dtype=np.uint8)

        ok, encoded = cv2.imencode(".png", vis)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return encoded.tobytes()

    def run(self):
        publish_dt = 1.0 / max(self.args.publish_hz, 1e-6)
        print_dt = 1.0 / max(self.args.print_hz, 1e-6)
        next_pub = time.monotonic()
        next_print = time.monotonic()

        last_frame_count = 0

        try:
            while True:
                now = time.monotonic()

                if now >= next_pub:
                    next_pub += publish_dt
                    with self._lock:
                        depth = None if self._latest_depth is None else self._latest_depth.copy()
                        scale = self._latest_scale
                        stamp_t = self._latest_stamp
                        depth_width = int(self._latest_depth_width)
                        depth_height = int(self._latest_depth_height)
                        depth_fx = float(self._latest_depth_fx)
                        depth_fy = float(self._latest_depth_fy)
                        depth_cx = float(self._latest_depth_cx)
                        depth_cy = float(self._latest_depth_cy)
                        frames = self._frames
                        robot_pos = self._latest_robot_pos.copy()
                        robot_yaw = float(self._latest_robot_yaw)
                        robot_quat_xyzw = self._latest_robot_quat_xyzw.copy()
                        target_pos = self._latest_target_pos.copy()
                        cmd_body = self._latest_cmd_body.copy()

                    if depth is not None and scale is not None and stamp_t is not None:
                        png_bytes = self._encode_depth_png(depth, scale)
                        b64 = base64.b64encode(png_bytes).decode("ascii")
                        stamp = self._to_stamp(stamp_t)

                        self.ch_image.log(
                            {
                                "timestamp": stamp,
                                "frame_id": "depth_noisy_viz",
                                "format": "png",
                                "data": b64,
                            }
                        )

                        # Publish 3D transforms for Foxglove 3D panel.
                        cy = math.cos(robot_yaw)
                        sy = math.sin(robot_yaw)
                        cmd_vx_w = cy * float(cmd_body[0]) - sy * float(cmd_body[1])
                        cmd_vy_w = sy * float(cmd_body[0]) + cy * float(cmd_body[1])
                        cmd_speed_w = math.sqrt(cmd_vx_w * cmd_vx_w + cmd_vy_w * cmd_vy_w)
                        cmd_yaw_w = math.atan2(cmd_vy_w, cmd_vx_w) if (cmd_vx_w * cmd_vx_w + cmd_vy_w * cmd_vy_w) > 1e-12 else robot_yaw

                        self.ch_tf_base.log(
                            self._tf_msg(stamp, "go2_base", robot_pos, robot_quat_xyzw)
                        )
                        self.ch_tf_target.log(
                            self._tf_msg(stamp, "go2_target", target_pos, [0.0, 0.0, 0.0, 1.0])
                        )

                        cmd_quat = self._yaw_to_quat_xyzw(cmd_yaw_w)
                        if self.ch_cmd_pose is not None:
                            self.ch_cmd_pose.log(
                                {
                                    "timestamp": stamp,
                                    "frame_id": "world",
                                    "pose": {
                                        "position": {
                                            "x": float(robot_pos[0]),
                                            "y": float(robot_pos[1]),
                                            "z": float(robot_pos[2]),
                                        },
                                        "orientation": {
                                            "x": float(cmd_quat[0]),
                                            "y": float(cmd_quat[1]),
                                            "z": float(cmd_quat[2]),
                                            "w": float(cmd_quat[3]),
                                        },
                                    },
                                }
                            )

                        arrow_scale = float(self.args.cmd_arrow_scale)
                        arrow_len = max(
                            float(self.args.cmd_arrow_min_len),
                            min(float(self.args.cmd_arrow_max_len), cmd_speed_w * arrow_scale),
                        )
                        self.ch_cmd_arrow.log(
                            {
                                "deletions": [],
                                "entities": [
                                    {
                                        "id": "go2_cmd_dir_arrow",
                                        "timestamp": stamp,
                                        "frame_id": "world",
                                        "arrows": [
                                            {
                                                "pose": {
                                                    "position": {
                                                        "x": float(robot_pos[0]),
                                                        "y": float(robot_pos[1]),
                                                        "z": float(robot_pos[2]),
                                                    },
                                                    "orientation": {
                                                        "x": float(cmd_quat[0]),
                                                        "y": float(cmd_quat[1]),
                                                        "z": float(cmd_quat[2]),
                                                        "w": float(cmd_quat[3]),
                                                    },
                                                },
                                                "shaft_length": float(arrow_len),
                                                "shaft_diameter": float(self.args.cmd_arrow_shaft_diameter),
                                                "head_length": float(self.args.cmd_arrow_head_length),
                                                "head_diameter": float(self.args.cmd_arrow_head_diameter),
                                                "color": {"r": 1.0, "g": 0.35, "b": 0.1, "a": 0.95},
                                            }
                                        ],
                                    }
                                ],
                            }
                        )

                        if (
                            self.ch_pointcloud is not None
                            and depth_width > 0
                            and depth_height > 0
                            and depth_fx > 0.0
                            and depth_fy > 0.0
                        ):
                            world_points = self._depth_to_world_points(
                                depth_u16=depth,
                                depth_scale=float(scale),
                                fx=depth_fx,
                                fy=depth_fy,
                                cx=depth_cx,
                                cy=depth_cy,
                                robot_pos=robot_pos.astype(np.float32),
                                robot_quat_xyzw=robot_quat_xyzw.astype(np.float32),
                            )
                            if world_points.shape[0] > 0:
                                cloud_bytes = world_points.astype("<f4", copy=False).tobytes()
                                cloud_b64 = base64.b64encode(cloud_bytes).decode("ascii")
                                self.ch_pointcloud.log(
                                    {
                                        "timestamp": stamp,
                                        "frame_id": "world",
                                        "pose": {
                                            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                                        },
                                        "point_stride": 12,
                                        "fields": [
                                            {"name": "x", "offset": 0, "type": 7},
                                            {"name": "y", "offset": 4, "type": 7},
                                            {"name": "z", "offset": 8, "type": 7},
                                        ],
                                        "data": cloud_b64,
                                    }
                                )

                if now >= next_print:
                    next_print += print_dt
                    with self._lock:
                        frames = self._frames
                        has_frame = self._latest_depth is not None
                    delta = frames - last_frame_count
                    last_frame_count = frames
                    print(
                        f"[DepthBridge] frames={frames} (+{delta}) has_frame={has_frame} "
                        f"ws=ws://{self.foxglove_host}:{self.foxglove_server.port}"
                    )

                time.sleep(0.001)
        except KeyboardInterrupt:
            print("[DepthBridge] Stopped by user.")
        finally:
            self.close()

    def close(self):
        for ch in [self.ch_image, self.ch_tf_base, self.ch_tf_target, self.ch_cmd_pose, self.ch_cmd_arrow, self.ch_pointcloud]:
            try:
                if ch is not None:
                    ch.close()
            except Exception:
                pass
        try:
            self.foxglove_server.stop()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subscribe depth DDS and broadcast to LAN via Foxglove WebSocket"
    )
    parser.add_argument("net", type=str, help="DDS network interface for ChannelFactoryInitialize, e.g. wlp3s0")
    parser.add_argument("--depth-topic", type=str, default="debug/depth_image_noisy", help="DDS depth topic")
    parser.add_argument("--pose-source", type=str, choices=["sport", "inekf"], default="sport", help="Robot pose source for base transform")
    parser.add_argument("--lowstate-topic", type=str, default="rt/lowstate", help="DDS lowstate topic (IMU orientation source)")
    parser.add_argument("--sport-topic", type=str, default="rt/sportmodestate", help="DDS robot state topic")
    parser.add_argument("--inekf-odom-topic", type=str, default="rt/inekf/odom", help="DDS INEKF odometry topic")
    parser.add_argument("--nav-target-topic", type=str, default="rt/nav_target", help="DDS nav target topic")
    parser.add_argument("--nav-debug-topic", type=str, default="rt/nav_debug", help="DDS nav debug topic")

    parser.add_argument("--foxglove-host", type=str, default="0.0.0.0", help="Foxglove bind host (ignored if --foxglove-interface is set)")
    parser.add_argument("--foxglove-interface", type=str, default="wlp3s0", help="Bind Foxglove WS to this NIC IPv4, e.g. wlp3s0")
    parser.add_argument("--foxglove-port", type=int, default=8765, help="Foxglove WS port (0 for auto)")

    parser.add_argument("--image-channel", type=str, default="/go2/depth/image", help="Foxglove channel for compressed image")
    parser.add_argument("--tf-base-channel", type=str, default="/go2/tf/base", help="Foxglove channel for robot base transform")
    parser.add_argument("--tf-target-channel", type=str, default="/go2/tf/target", help="Foxglove channel for target transform")
    parser.add_argument("--enable-cmd-pose", action="store_true", help="Enable legacy command direction pose topic")
    parser.add_argument("--cmd-pose-topic", type=str, default="/go2/cmd_dir", help="Foxglove topic for command direction pose")
    parser.add_argument("--cmd-arrow-topic", type=str, default="/go2/cmd_dir_arrow", help="Foxglove topic for dynamic command arrow")
    parser.add_argument("--cmd-arrow-scale", type=float, default=0.8, help="Arrow length scale (meter per m/s)")
    parser.add_argument("--cmd-arrow-min-len", type=float, default=0.08, help="Minimum arrow length in meters")
    parser.add_argument("--cmd-arrow-max-len", type=float, default=1.2, help="Maximum arrow length in meters")
    parser.add_argument("--cmd-arrow-shaft-diameter", type=float, default=0.03, help="Arrow shaft diameter")
    parser.add_argument("--cmd-arrow-head-length", type=float, default=0.08, help="Arrow head length")
    parser.add_argument("--cmd-arrow-head-diameter", type=float, default=0.06, help="Arrow head diameter")
    parser.add_argument("--pointcloud-topic", type=str, default="/go2/depth/points", help="Foxglove topic for depth point cloud")
    parser.add_argument("--no-pointcloud", action="store_true", help="Disable depth point cloud publishing")
    parser.add_argument("--pointcloud-downsample", type=int, default=2, help="Point cloud pixel stride")
    parser.add_argument("--pointcloud-max-points", type=int, default=20000, help="Point cloud max points per frame")
    parser.add_argument("--pointcloud-min-depth", type=float, default=0.25, help="Point cloud min depth in meters")
    parser.add_argument("--pointcloud-max-depth", type=float, default=10.0, help="Point cloud max depth in meters")

    parser.add_argument("--publish-hz", type=float, default=15.0, help="Foxglove publish rate")
    parser.add_argument("--print-hz", type=float, default=1.0, help="Console print rate")
    parser.add_argument("--visualize-min-depth", type=float, default=0.25, help="Depth min for PNG normalization")
    parser.add_argument("--visualize-max-depth", type=float, default=10.0, help="Depth max for PNG normalization")
    return parser.parse_args()


def main():
    args = parse_args()
    bridge = DepthBridge(args)
    bridge.run()


if __name__ == "__main__":
    main()
