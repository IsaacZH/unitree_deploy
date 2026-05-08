import argparse
import os
import socket
import sys
import threading
import time
from typing import Dict

DEPLOY_REAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEPLOY_REAL_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_REAL_DIR)

from unitree_sdk2py.core.channel import (  # type: ignore
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from bridge.lan_dds_bridge_common import (
    MSG_ACK,
    MSG_DATA,
    MSG_HEARTBEAT,
    MSG_RESUME,
    SafeSocketSender,
    TOPIC_BASE_POSE,
    TOPIC_CONTROL,
    TOPIC_DEPTH,
    TOPIC_NAME_TO_ID,
    TOPIC_NAV_DEBUG,
    TOPIC_NAV_TARGET,
    TopicReplayBuffer,
    deserialize_dds_message,
    pack_ack,
    pack_data,
    pack_heartbeat,
    recv_frame,
    serialize_dds_message,
)
from common.depth_image_sub import DepthImage_
from common.nav_debug_dds import NavDebug_
from common.nav_target_dds import NavTarget_
from keyboard.keyboard_command_dds import KeyboardCommand_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as OdometryGeo


class RobotBridge:
    def __init__(self, args):
        self.args = args
        self.sender = SafeSocketSender()

        self._seq = {
            TOPIC_BASE_POSE: 0,
            TOPIC_NAV_TARGET: 0,
            TOPIC_NAV_DEBUG: 0,
            TOPIC_DEPTH: 0,
        }

        self._replay = {
            TOPIC_BASE_POSE: TopicReplayBuffer(args.replay_base_pose),
            TOPIC_NAV_TARGET: TopicReplayBuffer(args.replay_nav),
            TOPIC_NAV_DEBUG: TopicReplayBuffer(args.replay_nav),
            TOPIC_DEPTH: TopicReplayBuffer(args.replay_depth),
        }

        self._conn_lock = threading.Lock()
        self._conn = None
        self._last_heartbeat = time.monotonic()
        self._last_depth_pub = 0.0

        ChannelFactoryInitialize(0, args.net)

        self.control_pub = ChannelPublisher(args.control_topic, KeyboardCommand_)
        self.control_pub.Init()

        self.base_pose_sub = ChannelSubscriber(args.base_pose_topic, OdometryGeo)
        self.base_pose_sub.Init(self._on_base_pose, 10)

        self.nav_target_sub = ChannelSubscriber(args.nav_target_topic, NavTarget_)
        self.nav_target_sub.Init(self._on_nav_target, 10)

        self.nav_debug_sub = ChannelSubscriber(args.nav_debug_topic, NavDebug_)
        self.nav_debug_sub.Init(self._on_nav_debug, 10)

        self.depth_sub = ChannelSubscriber(args.depth_topic, DepthImage_)
        self.depth_sub.Init(self._on_depth, 10)

    def _set_conn(self, conn):
        with self._conn_lock:
            self._conn = conn

    def _get_conn(self):
        with self._conn_lock:
            return self._conn

    def _send_data(self, topic_id: int, msg, compress: bool = False):
        conn = self._get_conn()
        if conn is None:
            return
        try:
            payload = serialize_dds_message(msg)
            self._seq[topic_id] += 1
            seq = self._seq[topic_id]
            self._replay[topic_id].append(seq=seq, payload=payload, compress=compress)
            frame = pack_data(topic_id=topic_id, seq=seq, payload=payload, compress=compress)
            self.sender.send(conn, frame)
        except Exception:
            self._drop_conn()

    def _drop_conn(self):
        conn = self._get_conn()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        self._set_conn(None)

    def _on_base_pose(self, msg: OdometryGeo):
        self._send_data(TOPIC_BASE_POSE, msg, compress=False)

    def _on_nav_target(self, msg: NavTarget_):
        self._send_data(TOPIC_NAV_TARGET, msg, compress=False)

    def _on_nav_debug(self, msg: NavDebug_):
        self._send_data(TOPIC_NAV_DEBUG, msg, compress=False)

    def _on_depth(self, msg: DepthImage_):
        now = time.monotonic()
        if now - self._last_depth_pub < (1.0 / max(self.args.depth_hz, 1e-6)):
            return
        self._last_depth_pub = now
        self._send_data(TOPIC_DEPTH, msg, compress=self.args.compress_depth)

    def _handle_frame(self, frame):
        msg_type = frame["msg_type"]

        if msg_type == MSG_HEARTBEAT:
            self._last_heartbeat = time.monotonic()
            conn = self._get_conn()
            if conn is not None:
                self.sender.send(conn, pack_heartbeat())
            return

        if msg_type == MSG_RESUME:
            self._replay_from_resume(frame["resume"])
            return

        if msg_type == MSG_DATA and frame["topic_id"] == TOPIC_CONTROL:
            msg = deserialize_dds_message(KeyboardCommand_, frame["payload"])
            self.control_pub.Write(msg)
            conn = self._get_conn()
            if conn is not None:
                self.sender.send(conn, pack_ack(TOPIC_CONTROL, frame["seq"]))
            return

        if msg_type == MSG_ACK:
            return

    def _replay_from_resume(self, last_seq_by_topic: Dict[int, int]):
        conn = self._get_conn()
        if conn is None:
            return
        for topic_id in (TOPIC_BASE_POSE, TOPIC_NAV_TARGET, TOPIC_NAV_DEBUG, TOPIC_DEPTH):
            last_seq = int(last_seq_by_topic.get(topic_id, 0))
            entries = self._replay[topic_id].entries_after(last_seq)
            for e in entries:
                frame = pack_data(topic_id=topic_id, seq=e.seq, payload=e.payload, compress=e.compress)
                self.sender.send(conn, frame)

    def _recv_loop(self, conn: socket.socket):
        try:
            while True:
                frame = recv_frame(conn)
                self._handle_frame(frame)
        except Exception:
            self._drop_conn()

    def _heartbeat_loop(self):
        while True:
            time.sleep(1.0)
            conn = self._get_conn()
            if conn is None:
                continue
            try:
                self.sender.send(conn, pack_heartbeat())
            except Exception:
                self._drop_conn()
                continue
            if time.monotonic() - self._last_heartbeat > self.args.heartbeat_timeout:
                self._drop_conn()

    def run(self):
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.args.host, self.args.port))
        server.listen(1)

        print(f"[RobotBridge] listening on {self.args.host}:{self.args.port}")

        while True:
            conn, addr = server.accept()
            print(f"[RobotBridge] client connected: {addr}")
            self._set_conn(conn)
            self._last_heartbeat = time.monotonic()
            t = threading.Thread(target=self._recv_loop, args=(conn,), daemon=True)
            t.start()


def parse_args():
    p = argparse.ArgumentParser(description="Robot-side DDS bridge server")
    p.add_argument("net", type=str, help="DDS network interface")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=16789)

    p.add_argument("--control-topic", type=str, default="rt/wireless_remote")
    p.add_argument("--base-pose-topic", type=str, default="rt/base_pose")
    p.add_argument("--nav-target-topic", type=str, default="rt/nav_target")
    p.add_argument("--nav-debug-topic", type=str, default="rt/nav_debug")
    p.add_argument("--depth-topic", type=str, default="debug/depth_image_noisy")

    p.add_argument("--depth-hz", type=float, default=12.0)
    p.add_argument("--compress-depth", action="store_true")

    p.add_argument("--replay-base-pose", type=int, default=256)
    p.add_argument("--replay-nav", type=int, default=256)
    p.add_argument("--replay-depth", type=int, default=64)
    p.add_argument("--heartbeat-timeout", type=float, default=3.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bridge = RobotBridge(args)
    bridge.run()
