"""CLI helper: publish /ee_target_pose from x y z yaw_deg.

Usage:
    ros2 run phy send_target -- 0.3 0.0 0.6 45
    ros2 run phy send_target -- 0.3 0.0 0.6        # yaw defaults to 0
"""

from __future__ import annotations

import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Yaw (rotation about world Z) → (x, y, z, w) unit quaternion."""
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Node("send_target")

    argv = sys.argv[1:]
    if len(argv) < 3:
        node.get_logger().error("usage: send_target -- x y z [yaw_deg]")
        rclpy.shutdown()
        return

    x, y, z = float(argv[0]), float(argv[1]), float(argv[2])
    yaw_deg = float(argv[3]) if len(argv) >= 4 else 0.0
    yaw_rad = math.radians(yaw_deg)

    qx, qy, qz, qw = _yaw_to_quaternion(yaw_rad)

    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw

    pub = node.create_publisher(PoseStamped, "/ee_target_pose", 10)

    # spin briefly so the publisher has time to connect before publishing
    import time
    rclpy.spin_once(node, timeout_sec=0.3)
    time.sleep(0.1)
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info(
        f"published target: xyz=({x}, {y}, {z}) yaw={yaw_deg:.1f}° ({yaw_rad:.4f} rad)"
    )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
