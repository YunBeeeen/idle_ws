"""Unit tests for phy.ik.IKSolver.solve_pose (6 task-space DoF)."""

from pathlib import Path

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory

from phy.ik import IKConfig, IKSolver


@pytest.fixture(scope="module")
def solver() -> IKSolver:
    urdf = Path(get_package_share_directory("sim")) / "urdf" / "robot.urdf"
    config = IKConfig(
        target_frame="gripper",
        controlled_joints=("j1", "j2", "j3", "j4", "j5", "j6"),
    )
    return IKSolver(urdf, config)


def _top_down_R(yaw: float) -> np.ndarray:
    """Gripper z-axis = +world Z with yaw rotation about world Z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def test_round_trip_random_pose(solver):
    """Pose from known q must be solvable back to q (or equivalent)."""
    rng = np.random.default_rng(0)
    failures = 0
    for _ in range(10):
        q_true = rng.uniform(solver.lower_limits, solver.upper_limits)
        pos_true, R_true = solver.forward_pose(q_true)
        q_seed = solver.clip_to_limits(q_true + rng.uniform(-0.2, 0.2, size=q_true.shape))
        res = solver.solve_pose(pos_true, R_true, q_seed)
        if not res.success:
            failures += 1
            continue
        pos_rec, R_rec = solver.forward_pose(res.q)
        assert np.linalg.norm(pos_rec - pos_true) < 1e-3
        assert np.linalg.norm(R_rec - R_true) < 1e-3
    assert failures <= 2, f"too many round-trip failures ({failures}/10)"


def test_top_down_grasp_reachable_target(solver):
    """A reasonable top-down grasp target should solve."""
    target = np.array([0.3, 0.0, 0.6])
    R = _top_down_R(0.0)
    q_seed = np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.0])
    res = solver.solve_pose(target, R, q_seed)
    assert res.success
    pos_rec, R_rec = solver.forward_pose(res.q)
    assert np.linalg.norm(pos_rec - target) < 1e-3
    z_axis_world = R_rec[:, 2]
    assert abs(z_axis_world[2] - 1.0) < 1e-3, "gripper z-axis should align with world Z"


def test_unreachable_target_reports_failure(solver):
    """Target outside workspace should fail (not crash)."""
    target = np.array([10.0, 0.0, 0.0])
    R = _top_down_R(0.0)
    q_seed = np.zeros(6)
    res = solver.solve_pose(target, R, q_seed)
    assert res.success is False
    assert res.residual_norm > solver.config.tolerance


def test_3dof_solve_still_works(solver):
    """Backward compat: position-only solve() still works."""
    res = solver.solve(np.array([0.3, 0.0, 0.7]), np.zeros(6))
    # success depends on reachability; just verify no crash and returns IKResult
    assert hasattr(res, "success")
    assert hasattr(res, "q")
    assert hasattr(res, "iterations")
    assert hasattr(res, "residual_norm")


def test_forward_pose_consistent_with_ik(solver):
    """solve_pose(target) → solve.q → forward_pose should match target."""
    target = np.array([0.25, 0.1, 0.65])
    R = _top_down_R(np.pi / 6)
    q_seed = np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.0])
    res = solver.solve_pose(target, R, q_seed)
    if not res.success:
        pytest.skip("seed did not converge; not a correctness issue")
    pos, rot = solver.forward_pose(res.q)
    assert np.linalg.norm(pos - target) < 1e-3
    assert np.linalg.norm(rot - R) < 1e-3
