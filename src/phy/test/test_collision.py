"""Unit tests for phy.collision."""

from pathlib import Path

import pytest
from ament_index_python.packages import get_package_share_directory

from phy.robot_model import RobotModel
from phy.collision import CollisionChecker
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP


@pytest.fixture(scope="module")
def sim_share() -> Path:
    return Path(get_package_share_directory("sim"))


@pytest.fixture(scope="module")
def collision_checker(sim_share) -> CollisionChecker:
    urdf = sim_share / "urdf" / "robot.urdf"
    srdf = sim_share / "srdf" / "robot.srdf"
    rm = RobotModel(urdf, DEFAULT_MOTOR_JOINT_MAP)
    return CollisionChecker(rm, srdf_path=srdf, package_dirs=[str(sim_share.parent)])


def test_srdf_reduces_pairs(collision_checker):
    assert collision_checker.n_pairs_active < collision_checker.n_pairs_total


def test_neutral_pose_collision_free(collision_checker):
    q = {m: 0.0 for m in collision_checker.robot.ordered_motor_ids}
    assert collision_checker.check(q) is False


def test_cage_disabled_by_default(collision_checker):
    assert collision_checker.cage_enabled is False


def test_cage_toggle_and_collision(collision_checker):
    """Extreme pose should collide with cage when cage mode is on."""
    q = {1: 0.0, 2: -2.5, 3: 2.5, 4: -2.5, 5: -1.57, 6: 0.0}
    # cage off — no floor, so self-collision only (may or may not collide)
    collision_checker.set_cage_enabled(False)
    assert collision_checker.cage_enabled is False

    collision_checker.set_cage_enabled(True)
    assert collision_checker.cage_enabled is True
    assert collision_checker.check(q) is True  # arm extends outside cage

    collision_checker.set_cage_enabled(False)  # restore


def test_cage_colliding_pairs_diagnostic(collision_checker):
    q = {1: 0.0, 2: -2.5, 3: 2.5, 4: -2.5, 5: -1.57, 6: 0.0}
    collision_checker.set_cage_enabled(True)
    pairs = collision_checker.colliding_pairs(q)
    collision_checker.set_cage_enabled(False)  # restore
    assert len(pairs) >= 1
    for a, b in pairs:
        assert isinstance(a, str) and isinstance(b, str)
    assert any("cage" in a.lower() or "cage" in b.lower() for a, b in pairs)


def test_check_trajectory_no_collision(collision_checker):
    samples = [{m: 0.0 for m in collision_checker.robot.ordered_motor_ids} for _ in range(5)]
    any_coll, idx = collision_checker.check_trajectory(samples)
    assert any_coll is False
    assert idx == -1


def test_check_trajectory_with_collision_returns_index(collision_checker):
    motor_ids = collision_checker.robot.ordered_motor_ids
    good = {m: 0.0 for m in motor_ids}
    bad = {1: 0.0, 2: -2.5, 3: 2.5, 4: -2.5, 5: -1.57, 6: 0.0}
    samples = [good, good, bad, good]
    collision_checker.set_cage_enabled(True)
    any_coll, idx = collision_checker.check_trajectory(samples)
    collision_checker.set_cage_enabled(False)  # restore
    assert any_coll is True
    assert idx == 2


def test_missing_srdf_raises(sim_share):
    urdf = sim_share / "urdf" / "robot.urdf"
    rm = RobotModel(urdf, DEFAULT_MOTOR_JOINT_MAP)
    with pytest.raises(FileNotFoundError):
        CollisionChecker(rm, srdf_path="/nonexistent/path.srdf", package_dirs=[str(sim_share.parent)])
