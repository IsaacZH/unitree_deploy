import argparse
import base64
import fcntl
import json
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
        self._frames = 0
        self._plasma_cmap = cm.get_cmap("plasma")

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
        self.ch_image = foxglove.Channel(
            self.args.image_channel,
            message_encoding="json",
            schema=compressed_schema_obj,
        )

        print(
            "[DepthBridge] Foxglove WS started at "
            f"ws://{self.foxglove_host}:{self.foxglove_server.port}"
        )
        print(f"[DepthBridge] Image channel: {self.args.image_channel}")

    def _init_dds(self):
        ChannelFactoryInitialize(0, self.args.net)
        self.sub = ChannelSubscriber(self.args.depth_topic, DepthImage_)
        self.sub.Init(self._depth_handler, 10)
        print(f"[DepthBridge] Subscribed DDS topic: {self.args.depth_topic}")

    def _depth_handler(self, msg: DepthImage_):
        try:
            depth = _decode_depth_message(msg).copy()
            with self._lock:
                self._latest_depth = depth
                self._latest_scale = float(msg.depth_scale)
                self._latest_stamp = time.time()
                self._frames += 1
        except Exception as exc:
            print(f"[DepthBridge] Decode error: {exc}")

    @staticmethod
    def _to_stamp(ts: float):
        ns = int(ts * 1e9)
        return {"sec": int(ns // 1_000_000_000), "nsec": int(ns % 1_000_000_000)}

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
                        frames = self._frames

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
        try:
            self.ch_image.close()
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

    parser.add_argument("--foxglove-host", type=str, default="0.0.0.0", help="Foxglove bind host (ignored if --foxglove-interface is set)")
    parser.add_argument("--foxglove-interface", type=str, default="wlp3s0", help="Bind Foxglove WS to this NIC IPv4, e.g. wlp3s0")
    parser.add_argument("--foxglove-port", type=int, default=8765, help="Foxglove WS port (0 for auto)")

    parser.add_argument("--image-channel", type=str, default="/go2/depth/image", help="Foxglove channel for compressed image")

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
