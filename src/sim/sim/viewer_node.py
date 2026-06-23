"""MuJoCo viewer node: mirror /motor_state_array pose into model joints."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional
import xml.etree.ElementTree as et

import mujoco
import rclpy
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP, parse_motor_joint_map_json
from idle_common.paths import resolve_share_file
from idle_common.ros_params import declare_typed
from msgs.msg import MotorStateArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


def load_model_with_workaround(model_xml: str) -> tuple[mujoco.MjModel, bool]:
    """Load model and patch inertial quat if needed for legacy fullinertia XML."""

    try:
        return mujoco.MjModel.from_xml_path(model_xml), False
    except ValueError as exc:
        if "fullinertia and inertial orientation" not in str(exc):
            raise

    tree = et.parse(model_xml)
    root = tree.getroot()
    modified = False
    for inertial in root.iter("inertial"):
        if "fullinertia" in inertial.attrib and "quat" in inertial.attrib:
            inertial.attrib.pop("quat")
            modified = True
    if not modified:
        raise ValueError("model failed to load and workaround found no inertial quat entries")

    patched_path: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix="mujoco_patched_",
            dir=str(Path(model_xml).parent),
            delete=False,
        ) as tmp:
            patched_path = tmp.name
            tree.write(tmp, encoding="unicode")
        return mujoco.MjModel.from_xml_path(patched_path), True
    finally:
        if patched_path is not None:
            Path(patched_path).unlink(missing_ok=True)


class ViewerNode(Node):
    """Update MuJoCo joint state from MotorStateArray."""

    def __init__(self) -> None:
        super().__init__("viewer_node")

        strip_str = lambda v: str(v).strip()

        self.viewer_hz = declare_typed(self, "viewer_hz", 60.0)
        self.viewer_enabled = declare_typed(self, "viewer", True)
        self.viewer_left_ui = declare_typed(self, "viewer_left_ui", True)
        self.viewer_right_ui = declare_typed(self, "viewer_right_ui", True)
        model_xml_text = declare_typed(self, "model_xml", "", cast=strip_str)
        map_json_text = declare_typed(self, "motor_joint_map_json", json.dumps(DEFAULT_MOTOR_JOINT_MAP))

        motor_joint_map = parse_motor_joint_map_json(map_json_text)
        if not motor_joint_map:
            motor_joint_map = dict(DEFAULT_MOTOR_JOINT_MAP)

        model_xml = resolve_share_file("sim", "robot.xml", model_xml_text)
        self.model, used_workaround = load_model_with_workaround(str(model_xml))
        self.data = mujoco.MjData(self.model)
        self.viewer: Optional[object] = None

        self.qpos_idx_by_motor: dict[int, int] = {}
        self.qvel_idx_by_motor: dict[int, int] = {}
        for motor_id, joint_name in sorted(motor_joint_map.items()):
            joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            if joint_id < 0:
                raise ValueError(f"joint not found in MuJoCo model: {joint_name} (motor_id={motor_id})")
            self.qpos_idx_by_motor[motor_id] = int(self.model.jnt_qposadr[joint_id])
            self.qvel_idx_by_motor[motor_id] = int(self.model.jnt_dofadr[joint_id])

        self.latest_q: dict[int, float] = {}
        self.latest_qd: dict[int, float] = {}
        self._state_dirty: bool = False

        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.state_sub = self.create_subscription(
            MotorStateArray, "/motor_state_array", self.on_state_array, qos_state
        )

        period_s = max(1.0 / self.viewer_hz, 1.0e-4)
        self.timer = self.create_timer(period_s, self.on_timer)
        self._try_launch_viewer()
        if used_workaround:
            self.get_logger().warn("model required inertial-quat workaround for fullinertia compatibility")
        self.get_logger().info(
            f"viewer_node initialized: model={model_xml} hz={self.viewer_hz:.1f} motors={sorted(self.qpos_idx_by_motor.keys())}"
        )

    def _try_launch_viewer(self) -> None:
        if not self.viewer_enabled:
            return
        try:
            import mujoco.viewer as mj_viewer

            self.viewer = mj_viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=self.viewer_left_ui,
                show_right_ui=self.viewer_right_ui,
            )
            self.get_logger().info("mujoco viewer launched")
        except Exception as exc:  # pragma: no cover - GUI runtime dependent
            self.viewer = None
            self.get_logger().warn(f"failed to launch mujoco viewer; running headless: {exc}")

    def on_state_array(self, msg: MotorStateArray) -> None:
        for state in msg.states:
            motor_id = int(state.motor_id)
            if motor_id not in self.qpos_idx_by_motor:
                continue
            self.latest_q[motor_id] = float(state.q)
            self.latest_qd[motor_id] = float(state.qd)
        self._state_dirty = True

    def on_timer(self) -> None:
        if self._state_dirty:
            for motor_id, qpos_idx in self.qpos_idx_by_motor.items():
                if motor_id in self.latest_q:
                    self.data.qpos[qpos_idx] = self.latest_q[motor_id]
                if motor_id in self.latest_qd:
                    self.data.qvel[self.qvel_idx_by_motor[motor_id]] = self.latest_qd[motor_id]
            mujoco.mj_forward(self.model, self.data)
            self._state_dirty = False

        if self.viewer is not None:
            try:
                self.viewer.sync()
                if not self.viewer.is_running():
                    self.viewer.close()
                    self.viewer = None
                    self.get_logger().warn("mujoco viewer closed; node continues headless")
            except Exception as exc:  # pragma: no cover - GUI runtime dependent
                self.get_logger().warn(f"viewer sync failed; switching to headless mode: {exc}")
                try:
                    self.viewer.close()
                except Exception:
                    pass
                self.viewer = None

    def destroy_node(self) -> bool:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ViewerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
