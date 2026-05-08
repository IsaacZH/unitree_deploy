import collections
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

MAGIC = b"LDB1"

MSG_DATA = 1
MSG_ACK = 2
MSG_RESUME = 3
MSG_HEARTBEAT = 4

FLAG_COMPRESSED = 1 << 0

TOPIC_CONTROL = 1
TOPIC_BASE_POSE = 2
TOPIC_NAV_TARGET = 3
TOPIC_NAV_DEBUG = 4
TOPIC_DEPTH = 5

TOPIC_NAME_TO_ID = {
    "rt/wireless_remote": TOPIC_CONTROL,
    "rt/base_pose": TOPIC_BASE_POSE,
    "rt/nav_target": TOPIC_NAV_TARGET,
    "rt/nav_debug": TOPIC_NAV_DEBUG,
    "debug/depth_image_noisy": TOPIC_DEPTH,
}

TOPIC_ID_TO_NAME = {v: k for k, v in TOPIC_NAME_TO_ID.items()}

HEADER = struct.Struct("!4sBHIQIB")
# magic, msg_type, topic_id, flags, seq, payload_len, resume_count

ACK_BODY = struct.Struct("!HQ")
# topic_id, seq

RESUME_ITEM = struct.Struct("!HQ")
# topic_id, seq

HEARTBEAT_BODY = struct.Struct("!Q")
# monotonic ms


def now_ms() -> int:
    return int(time.monotonic() * 1000.0)


def serialize_dds_message(msg) -> bytes:
    if hasattr(msg, "serialize"):
        return msg.serialize()
    raise TypeError(f"Message type does not support serialize(): {type(msg)}")


def deserialize_dds_message(msg_cls, payload: bytes):
    if hasattr(msg_cls, "deserialize"):
        return msg_cls.deserialize(payload)
    raise TypeError(f"Message class does not support deserialize(): {msg_cls}")


def pack_data(topic_id: int, seq: int, payload: bytes, compress: bool = False) -> bytes:
    flags = 0
    body = payload
    if compress:
        body = zlib.compress(payload, level=1)
        flags |= FLAG_COMPRESSED
    header = HEADER.pack(MAGIC, MSG_DATA, topic_id, flags, seq, len(body), 0)
    frame = header + body
    return struct.pack("!I", len(frame)) + frame


def pack_ack(topic_id: int, seq: int) -> bytes:
    body = ACK_BODY.pack(topic_id, seq)
    header = HEADER.pack(MAGIC, MSG_ACK, topic_id, 0, seq, len(body), 0)
    frame = header + body
    return struct.pack("!I", len(frame)) + frame


def pack_resume(last_seq_by_topic: Dict[int, int]) -> bytes:
    count = len(last_seq_by_topic)
    payload = bytearray()
    for topic_id, seq in last_seq_by_topic.items():
        payload.extend(RESUME_ITEM.pack(topic_id, seq))
    body = bytes(payload)
    header = HEADER.pack(MAGIC, MSG_RESUME, 0, 0, 0, len(body), count)
    frame = header + body
    return struct.pack("!I", len(frame)) + frame


def pack_heartbeat() -> bytes:
    body = HEARTBEAT_BODY.pack(now_ms())
    header = HEADER.pack(MAGIC, MSG_HEARTBEAT, 0, 0, 0, len(body), 0)
    frame = header + body
    return struct.pack("!I", len(frame)) + frame


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data.extend(chunk)
    return bytes(data)


def recv_frame(sock: socket.socket):
    frame_len = struct.unpack("!I", recv_exact(sock, 4))[0]
    frame = recv_exact(sock, frame_len)

    if len(frame) < HEADER.size:
        raise ValueError("Frame too short")

    magic, msg_type, topic_id, flags, seq, payload_len, resume_count = HEADER.unpack(
        frame[: HEADER.size]
    )
    if magic != MAGIC:
        raise ValueError("Invalid magic")

    payload = frame[HEADER.size : HEADER.size + payload_len]
    if len(payload) != payload_len:
        raise ValueError("Invalid payload length")

    if msg_type == MSG_DATA and (flags & FLAG_COMPRESSED):
        payload = zlib.decompress(payload)

    if msg_type == MSG_RESUME:
        expected = resume_count * RESUME_ITEM.size
        if payload_len != expected:
            raise ValueError("Invalid resume payload size")
        last_seq_by_topic: Dict[int, int] = {}
        off = 0
        for _ in range(resume_count):
            t_id, t_seq = RESUME_ITEM.unpack(payload[off : off + RESUME_ITEM.size])
            last_seq_by_topic[t_id] = t_seq
            off += RESUME_ITEM.size
        return {
            "msg_type": msg_type,
            "topic_id": topic_id,
            "flags": flags,
            "seq": seq,
            "payload": payload,
            "resume": last_seq_by_topic,
        }

    if msg_type == MSG_ACK:
        if payload_len != ACK_BODY.size:
            raise ValueError("Invalid ACK size")
        ack_topic, ack_seq = ACK_BODY.unpack(payload)
        return {
            "msg_type": msg_type,
            "topic_id": ack_topic,
            "flags": flags,
            "seq": ack_seq,
            "payload": payload,
        }

    return {
        "msg_type": msg_type,
        "topic_id": topic_id,
        "flags": flags,
        "seq": seq,
        "payload": payload,
    }


@dataclass
class ReplayEntry:
    seq: int
    payload: bytes
    compress: bool


class TopicReplayBuffer:
    def __init__(self, maxlen: int):
        self._buf: Deque[ReplayEntry] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, seq: int, payload: bytes, compress: bool) -> None:
        with self._lock:
            self._buf.append(ReplayEntry(seq=seq, payload=payload, compress=compress))

    def entries_after(self, seq: int) -> List[ReplayEntry]:
        with self._lock:
            return [e for e in self._buf if e.seq > seq]


class SafeSocketSender:
    def __init__(self):
        self._lock = threading.Lock()

    def send(self, sock: socket.socket, data: bytes) -> None:
        with self._lock:
            sock.sendall(data)
