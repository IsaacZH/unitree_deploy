import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

DEPLOY_REAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEPLOY_REAL_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_REAL_DIR)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    nav_msgs_msg_dds__Odometry_,
    geometry_msgs_msg_dds__PoseStamped_,
    geometry_msgs_msg_dds__Pose_,
    geometry_msgs_msg_dds__PoseWithCovarianceStamped_,
    unitree_go_msg_dds__SportModeState_,
)
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import (
    PoseStamped_ as PoseStampedGeo,
    Pose_ as PoseGeo,
    PoseWithCovarianceStamped_ as PoseWithCovarianceStampedGeo,
)
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as OdometryGeo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as SportModeStateGo


@dataclass
class StreamState:
    pos: np.ndarray
    count: int
    last_rx_monotonic: float


class PosePrinter:
    def __init__(self, args):
        self.args = args
        now = time.monotonic()
        self._sport = StreamState(np.array([np.nan, np.nan, np.nan], dtype=np.float32), 0, now)
        self._odom = StreamState(np.array([np.nan, np.nan, np.nan], dtype=np.float32), 0, now)
        self._robot_pose = StreamState(np.array([np.nan, np.nan, np.nan], dtype=np.float32), 0, now)
        self._lock = threading.Lock()

        self.sport_sub = ChannelSubscriber(args.sport_topic, SportModeStateGo)
        self.sport_sub.Init(self._on_sport, args.queue_depth)

        self.odom_sub = ChannelSubscriber(args.odom_topic, OdometryGeo)
        self.odom_sub.Init(self._on_odom, args.queue_depth)

        pose_msg_type = self._resolve_pose_msg_type(args.robot_pose_type)
        self.robot_pose_sub = ChannelSubscriber(args.robot_pose_topic, pose_msg_type)
        self.robot_pose_sub.Init(self._on_robot_pose, args.queue_depth)

    @staticmethod
    def _resolve_pose_msg_type(pose_type: str):
        mapping = {
            "pose_stamped": PoseStampedGeo,
            "pose": PoseGeo,
            "pose_with_cov_stamped": PoseWithCovarianceStampedGeo,
            "odom": OdometryGeo,
        }
        return mapping[pose_type]

    def _on_sport(self, msg: SportModeStateGo):
        pos = np.asarray(getattr(msg, "position", [np.nan, np.nan, np.nan]), dtype=np.float32).ravel()
        out = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        if pos.shape[0] >= 3:
            out[:] = pos[:3]
        with self._lock:
            self._sport.pos = out
            self._sport.count += 1
            self._sport.last_rx_monotonic = time.monotonic()

    def _on_odom(self, msg: OdometryGeo):
        out = np.array(
            [
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                float(msg.pose.pose.position.z),
            ],
            dtype=np.float32,
        )
        with self._lock:
            self._odom.pos = out
            self._odom.count += 1
            self._odom.last_rx_monotonic = time.monotonic()

    @staticmethod
    def _extract_pose_xyz(msg) -> np.ndarray:
        out = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        # Handle nav_msgs/Odometry: msg.pose.pose.position
        if hasattr(msg, "pose") and hasattr(msg.pose, "pose") and hasattr(msg.pose.pose, "position"):
            p = msg.pose.pose.position
            out[:] = [float(p.x), float(p.y), float(p.z)]
            return out

        # Handle geometry_msgs/PoseStamped: msg.pose.position
        if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
            p = msg.pose.position
            out[:] = [float(p.x), float(p.y), float(p.z)]
            return out

        # Handle geometry_msgs/Pose: msg.position
        if hasattr(msg, "position"):
            p = msg.position
            out[:] = [float(p.x), float(p.y), float(p.z)]
            return out

        return out

    def _on_robot_pose(self, msg):
        out = self._extract_pose_xyz(msg)
        with self._lock:
            self._robot_pose.pos = out
            self._robot_pose.count += 1
            self._robot_pose.last_rx_monotonic = time.monotonic()

    @staticmethod
    def _fmt_pos(pos: np.ndarray) -> str:
        return f"({pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f})"

    def run(self):
        print(
            "[PosePrinter] Subscribed "
            f"sport={self.args.sport_topic} odom={self.args.odom_topic} "
            f"robot_pose={self.args.robot_pose_topic}({self.args.robot_pose_type}) "
            f"print_hz={self.args.print_hz:.2f} stale_timeout={self.args.stale_timeout:.2f}s"
        )

        period = 1.0 / max(self.args.print_hz, 1.0e-6)
        no_data_warned = False

        while True:
            now_mono = time.monotonic()
            now_wall = time.strftime("%H:%M:%S", time.localtime())

            with self._lock:
                sport_pos = self._sport.pos.copy()
                sport_count = self._sport.count
                sport_age = now_mono - self._sport.last_rx_monotonic

                odom_pos = self._odom.pos.copy()
                odom_count = self._odom.count
                odom_age = now_mono - self._odom.last_rx_monotonic

                robot_pose_pos = self._robot_pose.pos.copy()
                robot_pose_count = self._robot_pose.count
                robot_pose_age = now_mono - self._robot_pose.last_rx_monotonic

            sport_stale = sport_age > self.args.stale_timeout
            odom_stale = odom_age > self.args.stale_timeout
            robot_pose_stale = robot_pose_age > self.args.stale_timeout

            extra = []
            if sport_stale:
                extra.append(f"sport_stale={sport_age:.2f}s")
            if odom_stale:
                extra.append(f"odom_stale={odom_age:.2f}s")
            if robot_pose_stale:
                extra.append(f"robot_pose_stale={robot_pose_age:.2f}s")
            suffix = " | " + ", ".join(extra) if extra else ""

            print(
                f"[{now_wall}] "
                f"sport_pos={self._fmt_pos(sport_pos)} (n={sport_count}) "
                f"odom_pos={self._fmt_pos(odom_pos)} (n={odom_count}) "
                f"robot_pose_pos={self._fmt_pos(robot_pose_pos)} (n={robot_pose_count})"
                f"{suffix}"
            )

            if not no_data_warned and (sport_count == 0 or odom_count == 0 or robot_pose_count == 0) and now_mono > 5.0:
                print(
                    "[PosePrinter] Waiting for data... "
                    f"check net={self.args.net}, sport_topic={self.args.sport_topic}, "
                    f"odom_topic={self.args.odom_topic}, robot_pose_topic={self.args.robot_pose_topic}"
                )
                no_data_warned = True

            time.sleep(period)

    def summary(self):
        with self._lock:
            return self._sport.count, self._odom.count, self._robot_pose.count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subscribe sportmode + odom + robot_pose DDS and print xyz positions."
    )
    parser.add_argument("net", type=str, help="network interface, e.g. eno1 or wlp3s0")
    parser.add_argument("--sport-topic", type=str, default="rt/sportmodestate")
    parser.add_argument("--odom-topic", type=str, default="rt/utlidar/robot_odom")
    parser.add_argument("--robot-pose-topic", type=str, default="rt/utlidar/robot_pose")
    parser.add_argument(
        "--robot-pose-type",
        type=str,
        choices=["pose_stamped", "pose", "pose_with_cov_stamped", "odom"],
        default="pose_stamped",
        help="message type for --robot-pose-topic",
    )
    parser.add_argument("--print-hz", type=float, default=10.0, help="print frequency")
    parser.add_argument("--queue-depth", type=int, default=10, help="DDS subscriber queue depth")
    parser.add_argument(
        "--stale-timeout",
        type=float,
        default=1.0,
        help="mark stream stale when no message for this many seconds",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ChannelFactoryInitialize(0, args.net)

    runner = PosePrinter(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        sport_n, odom_n, robot_pose_n = runner.summary()
        print("\n[PosePrinter] Exit.")
        print(
            "[PosePrinter] total "
            f"sport messages={sport_n}, odom messages={odom_n}, robot_pose messages={robot_pose_n}"
        )


if __name__ == "__main__":
    main()
