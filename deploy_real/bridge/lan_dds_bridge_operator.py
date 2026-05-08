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
    TOPIC_BASE_POSE,
    TOPIC_CONTROL,
    TOPIC_DEPTH,
    TOPIC_NAV_DEBUG,
    TOPIC_NAV_TARGET,
    deserialize_dds_message,
    pack_ack,
    pack_data,
    pack_heartbeat,
    pack_resume,
    recv_frame,
    serialize_dds_message,
    SafeSocketSender,
)
from common.depth_image_sub import DepthImage_
from common.nav_debug_dds import NavDebug_
from common.nav_target_dds import NavTarget_
from keyboard.keyboard_command_dds import KeyboardCommand_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as OdometryGeo


class LatestControlBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None

    def set(self, msg):
        with self._lock:
            self._latest = msg

    def pop(self):
        with self._lock:
            msg = self._latest
            self._latest = None
            return msg


class OperatorBridge:
    def __init__(self, args):
        self.args = args
        self.sender = SafeSocketSender()

        self._conn = None
        self._conn_lock = threading.Lock()

        self._control_seq = 0
        self._last_heartbeat = time.monotonic()
        self._last_recv_seq = {
            TOPIC_BASE_POSE: 0,
            TOPIC_NAV_TARGET: 0,
            TOPIC_NAV_DEBUG: 0,
            TOPIC_DEPTH: 0,
            TOPIC_CONTROL: 0,
        }

        self._latest_control = LatestControlBuffer()

        ChannelFactoryInitialize(0, args.net)

        self.control_sub = ChannelSubscriber(args.control_topic, KeyboardCommand_)
        self.control_sub.Init(self._on_control, 10)

        self.base_pose_pub = ChannelPublisher(args.base_pose_topic, OdometryGeo)
        self.base_pose_pub.Init()
        self.nav_target_pub = ChannelPublisher(args.nav_target_topic, NavTarget_)
        self.nav_target_pub.Init()
        self.nav_debug_pub = ChannelPublisher(args.nav_debug_topic, NavDebug_)
        self.nav_debug_pub.Init()
        self.depth_pub = ChannelPublisher(args.depth_topic, DepthImage_)
        self.depth_pub.Init()

    def _set_conn(self, conn):
        with self._conn_lock:
            self._conn = conn

    def _get_conn(self):
        with self._conn_lock:
            return self._conn

    def _drop_conn(self):
        conn = self._get_conn()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        self._set_conn(None)

    def _on_control(self, msg: KeyboardCommand_):
        self._latest_control.set(msg)

    def _publish_viz(self, topic_id: int, payload: bytes):
        if topic_id == TOPIC_BASE_POSE:
            self.base_pose_pub.Write(deserialize_dds_message(OdometryGeo, payload))
            return
        if topic_id == TOPIC_NAV_TARGET:
            self.nav_target_pub.Write(deserialize_dds_message(NavTarget_, payload))
            return
        if topic_id == TOPIC_NAV_DEBUG:
            self.nav_debug_pub.Write(deserialize_dds_message(NavDebug_, payload))
            return
        if topic_id == TOPIC_DEPTH:
            self.depth_pub.Write(deserialize_dds_message(DepthImage_, payload))
            return

    def _send_resume(self, conn: socket.socket):
        resume = {
            TOPIC_BASE_POSE: self._last_recv_seq[TOPIC_BASE_POSE],
            TOPIC_NAV_TARGET: self._last_recv_seq[TOPIC_NAV_TARGET],
            TOPIC_NAV_DEBUG: self._last_recv_seq[TOPIC_NAV_DEBUG],
            TOPIC_DEPTH: self._last_recv_seq[TOPIC_DEPTH],
        }
        self.sender.send(conn, pack_resume(resume))

    def _recv_loop(self, conn: socket.socket):
        try:
            while True:
                frame = recv_frame(conn)
                msg_type = frame["msg_type"]

                if msg_type == MSG_HEARTBEAT:
                    self._last_heartbeat = time.monotonic()
                    self.sender.send(conn, pack_heartbeat())
                    continue

                if msg_type == MSG_ACK:
                    self._last_recv_seq[TOPIC_CONTROL] = max(
                        self._last_recv_seq[TOPIC_CONTROL], frame["seq"]
                    )
                    continue

                if msg_type == MSG_DATA:
                    topic_id = frame["topic_id"]
                    seq = frame["seq"]
                    if seq <= self._last_recv_seq.get(topic_id, 0):
                        continue
                    self._last_recv_seq[topic_id] = seq
                    self._publish_viz(topic_id, frame["payload"])
                    self.sender.send(conn, pack_ack(topic_id, seq))
        except Exception:
            self._drop_conn()

    def _control_send_loop(self):
        while True:
            time.sleep(1.0 / max(self.args.control_send_hz, 1e-6))
            conn = self._get_conn()
            if conn is None:
                continue
            msg = self._latest_control.pop()
            if msg is None:
                continue
            try:
                payload = serialize_dds_message(msg)
                self._control_seq += 1
                frame = pack_data(
                    topic_id=TOPIC_CONTROL,
                    seq=self._control_seq,
                    payload=payload,
                    compress=False,
                )
                self.sender.send(conn, frame)
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
        threading.Thread(target=self._control_send_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

        retry_steps = [0.5, 1.0, 2.0, 4.0]

        while True:
            connected = False
            for wait_s in retry_steps:
                try:
                    conn = socket.create_connection((self.args.robot_host, self.args.robot_port), timeout=3.0)
                    conn.settimeout(None)
                    self._set_conn(conn)
                    self._last_heartbeat = time.monotonic()
                    self._send_resume(conn)
                    threading.Thread(target=self._recv_loop, args=(conn,), daemon=True).start()
                    print(f"[OperatorBridge] connected to {self.args.robot_host}:{self.args.robot_port}")
                    connected = True
                    break
                except Exception:
                    time.sleep(wait_s)

            if not connected:
                continue

            while self._get_conn() is not None:
                time.sleep(0.2)

            print("[OperatorBridge] disconnected, retrying...")


def parse_args():
    p = argparse.ArgumentParser(description="Operator-side DDS bridge client")
    p.add_argument("net", type=str, help="DDS network interface")
    p.add_argument("--robot-host", type=str, required=True)
    p.add_argument("--robot-port", type=int, default=16789)

    p.add_argument("--control-topic", type=str, default="rt/wireless_remote")
    p.add_argument("--base-pose-topic", type=str, default="rt/base_pose")
    p.add_argument("--nav-target-topic", type=str, default="rt/nav_target")
    p.add_argument("--nav-debug-topic", type=str, default="rt/nav_debug")
    p.add_argument("--depth-topic", type=str, default="debug/depth_image_noisy")

    p.add_argument("--control-send-hz", type=float, default=50.0)
    p.add_argument("--heartbeat-timeout", type=float, default=3.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bridge = OperatorBridge(args)
    bridge.run()
