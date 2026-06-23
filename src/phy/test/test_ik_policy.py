import numpy as np
import pytest

from phy.ik import IKPolicyConfig, IKResult, IKSolver


def _make_solver(dof: int = 3, lower: float = -np.pi, upper: float = np.pi) -> IKSolver:
    solver = object.__new__(IKSolver)
    solver.q_indices = list(range(dof))
    solver.lower_limits = np.full(dof, float(lower), dtype=float)
    solver.upper_limits = np.full(dof, float(upper), dtype=float)
    return solver


def _attach_scripted_solve(solver: IKSolver, scripted_results: list[IKResult]) -> None:
    state = {"idx": 0}

    def _solve(goal_xyz: np.ndarray, q_seed: np.ndarray) -> IKResult:
        del goal_xyz, q_seed
        idx = state["idx"]
        state["idx"] += 1
        if idx >= len(scripted_results):
            idx = len(scripted_results) - 1
        return scripted_results[idx]

    solver.solve = _solve  # type: ignore[method-assign]


def test_accept_reachable_goal_with_policy() -> None:
    solver = _make_solver(dof=3)
    scripted_results = [
        IKResult(False, np.array([0.7, 0.2, 0.1]), 100, 0.05),
        IKResult(True, np.array([0.23, 0.10, -0.10]), 12, 1.0e-4),
        IKResult(True, np.array([0.90, 0.50, 0.20]), 15, 5.0e-5),
        IKResult(True, np.array([0.35, 0.10, -0.10]), 10, 3.0e-4),
    ]
    _attach_scripted_solve(solver, scripted_results)

    policy = IKPolicyConfig(
        max_ik_residual_accept_m=0.02,
        ik_random_restarts=0,
        max_joint_jump_rad=1.2,
        use_heuristic_seed=True,
    )
    q_ref = np.array([0.20, 0.10, -0.10], dtype=float)
    result = solver.solve_with_policy(
        goal_xyz=np.array([0.20, 0.00, 0.35], dtype=float),
        q_ref=q_ref,
        q_measured=q_ref.copy(),
        policy=policy,
        rng=np.random.default_rng(0),
    )

    assert result.accepted
    assert result.reason == "accepted"
    assert result.seeds_tested == 4
    assert result.best_residual == pytest.approx(1.0e-4)
    assert result.q_goal is not None
    np.testing.assert_allclose(result.q_goal, np.array([0.23, 0.10, -0.10]), atol=1.0e-9)


def test_reject_when_residual_too_large() -> None:
    solver = _make_solver(dof=3)
    scripted_results = [
        IKResult(False, np.array([0.6, 0.0, 0.0]), 100, 0.08),
        IKResult(False, np.array([0.4, 0.0, 0.0]), 100, 0.03),
        IKResult(False, np.array([0.5, 0.0, 0.0]), 100, 0.04),
    ]
    _attach_scripted_solve(solver, scripted_results)

    policy = IKPolicyConfig(
        max_ik_residual_accept_m=0.02,
        ik_random_restarts=0,
        max_joint_jump_rad=1.2,
        use_heuristic_seed=True,
    )
    result = solver.solve_with_policy(
        goal_xyz=np.array([0.50, 0.00, 0.60], dtype=float),
        q_ref=np.zeros(3, dtype=float),
        q_measured=None,
        policy=policy,
        rng=np.random.default_rng(0),
    )

    assert not result.accepted
    assert result.reason == "rejected_residual"
    assert result.q_goal is None
    assert result.best_residual == pytest.approx(0.03)
    assert result.seeds_tested == 3


def test_reject_when_joint_jump_exceeds_limit() -> None:
    solver = _make_solver(dof=3)
    scripted_results = [
        IKResult(True, np.array([2.0, 0.0, 0.0]), 10, 1.0e-5),
        IKResult(True, np.array([2.1, 0.0, 0.0]), 10, 2.0e-5),
        IKResult(True, np.array([2.2, 0.0, 0.0]), 10, 3.0e-5),
    ]
    _attach_scripted_solve(solver, scripted_results)

    policy = IKPolicyConfig(
        max_ik_residual_accept_m=0.02,
        ik_random_restarts=0,
        max_joint_jump_rad=0.5,
        use_heuristic_seed=True,
    )
    result = solver.solve_with_policy(
        goal_xyz=np.array([0.20, 0.00, 0.35], dtype=float),
        q_ref=np.zeros(3, dtype=float),
        q_measured=None,
        policy=policy,
        rng=np.random.default_rng(0),
    )

    assert not result.accepted
    assert result.reason == "rejected_joint_jump"
    assert result.q_goal is None
    assert result.best_jump_rad > 0.5


def test_prefers_solution_closest_to_reference() -> None:
    solver = _make_solver(dof=3)
    scripted_results = [
        IKResult(True, np.array([0.8, 0.0, 0.0]), 8, 1.0e-6),
        IKResult(True, np.array([0.2, 0.0, 0.0]), 9, 1.0e-3),
        IKResult(True, np.array([1.5, 0.0, 0.0]), 7, 1.0e-7),
    ]
    _attach_scripted_solve(solver, scripted_results)

    policy = IKPolicyConfig(
        max_ik_residual_accept_m=0.02,
        ik_random_restarts=0,
        max_joint_jump_rad=2.0,
        use_heuristic_seed=True,
    )
    result = solver.solve_with_policy(
        goal_xyz=np.array([0.25, 0.00, 0.35], dtype=float),
        q_ref=np.zeros(3, dtype=float),
        q_measured=None,
        policy=policy,
        rng=np.random.default_rng(0),
    )

    assert result.accepted
    assert result.q_goal is not None
    np.testing.assert_allclose(result.q_goal, np.array([0.2, 0.0, 0.0]), atol=1.0e-9)
    assert result.best_residual == pytest.approx(1.0e-3)
