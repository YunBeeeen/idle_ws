"""Unit tests for phy.robot_model."""

from pathlib import Path

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory

from phy.robot_model import RobotModel
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP


@pytest.fixture(scope="module")
def urdf_path() -> Path:
    return Path(get_package_share_directory("sim")) / "urdf" / "robot.urdf"


@pytest.fixture(scope="module")
def robot(urdf_path) -> RobotModel:
    return RobotModel(urdf_path, DEFAULT_MOTOR_JOINT_MAP)


def test_motors_match_motor_map(robot):
    assert robot.ordered_motor_ids == tuple(sorted(DEFAULT_MOTOR_JOINT_MAP.keys()))
    for motor_id, joint_name in DEFAULT_MOTOR_JOINT_MAP.items():
        assert robot.bindings[motor_id].joint_name == joint_name


def test_neutral_pose_gravity_is_small(robot):
    """At neutral (arm straight up), gravity torques should be near zero."""
    q = {m: 0.0 for m in robot.ordered_motor_ids}
    tau = robot.gravity_torque(q)
    for motor_id, value in tau.items():
        assert abs(value) < 0.01, f"motor {motor_id}: |tau| should be small at neutral, got {value}"


def test_horizontal_pose_gravity_nonzero(robot):
    """With arm out (j2=π/2), shoulder gravity should be nonzero."""
    q = {1: 0.0, 2: 1.5, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
    tau = robot.gravity_torque(q)
    assert abs(tau[2]) > 1.0


def test_forward_kinematics_returns_se3(robot):
    q = {m: 0.0 for m in robot.ordered_motor_ids}
    se3 = robot.forward_kinematics(q, "gripper")
    assert se3.translation.shape == (3,)
    assert se3.rotation.shape == (3, 3)


def test_jacobian_shape(robot):
    q = {m: 0.0 for m in robot.ordered_motor_ids}
    J = robot.jacobian(q, "gripper")
    n = len(robot.ordered_motor_ids)
    assert J.shape == (6, n)


def test_mass_matrix_symmetric_and_positive(robot):
    q = {m: 0.1 for m in robot.ordered_motor_ids}
    M = robot.mass_matrix(q)
    n = len(robot.ordered_motor_ids)
    assert M.shape == (n, n)
    assert np.allclose(M, M.T, atol=1e-9), "mass matrix must be symmetric"
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0), "mass matrix must be positive definite"


def test_joint_limits_shape_and_order(robot):
    lower, upper = robot.joint_limits()
    n = len(robot.ordered_motor_ids)
    assert lower.shape == (n,)
    assert upper.shape == (n,)
    assert np.all(upper > lower)


def test_missing_motor_raises(robot):
    q = {m: 0.0 for m in robot.ordered_motor_ids}
    del q[robot.ordered_motor_ids[0]]
    with pytest.raises(KeyError):
        robot.gravity_torque(q)


def test_unknown_frame_raises(robot):
    q = {m: 0.0 for m in robot.ordered_motor_ids}
    with pytest.raises(ValueError, match="frame not found"):
        robot.forward_kinematics(q, "nonexistent_frame")


def test_gripper_width_not_yet_implemented(robot):
    with pytest.raises(NotImplementedError):
        robot.gripper_width(0.5)
