import numpy as np

from phy.ik import IKPolicyConfig, IKResult, IKSolver


def _make_solver(dof: int = 3, lower: float = -np.pi, upper: float = np.pi) -> IKSolver:
    solver = object.__new__(IKSolver)
    solver.q_indices = list(range(dof))
    solver.lower_limits = np.full(dof, float(lower), dtype=float)
    solver.upper_limits = np.full(dof, float(upper), dtype=float)
    return solver


def test_policy_is_stable_for_repeated_goal_with_fixed_rng() -> None:
    solver = _make_solver(dof=3)

    def _solve(goal_xyz: np.ndarray, q_seed: np.ndarray) -> IKResult:
        del goal_xyz
        q = np.tanh(np.asarray(q_seed, dtype=float))
        residual = float(np.linalg.norm(q - np.array([0.2, 0.0, 0.35], dtype=float)))
        return IKResult(success=(residual < 0.2), q=q, iterations=7, residual_norm=residual)

    solver.solve = _solve  # type: ignore[method-assign]
    policy = IKPolicyConfig(
        max_ik_residual_accept_m=0.5,
        ik_random_restarts=8,
        ik_seed_default_span=np.pi,
        max_joint_jump_rad=2.0,
        use_heuristic_seed=True,
    )
    goal = np.array([0.2, 0.0, 0.35], dtype=float)
    q_ref = np.array([0.15, -0.05, 0.10], dtype=float)

    first = solver.solve_with_policy(
        goal_xyz=goal,
        q_ref=q_ref,
        q_measured=None,
        policy=policy,
        rng=np.random.default_rng(0),
    )
    second = solver.solve_with_policy(
        goal_xyz=goal,
        q_ref=q_ref,
        q_measured=None,
        policy=policy,
        rng=np.random.default_rng(0),
    )

    assert first.accepted
    assert second.accepted
    assert first.reason == "accepted"
    assert second.reason == "accepted"
    assert first.q_goal is not None
    assert second.q_goal is not None
    np.testing.assert_allclose(first.q_goal, second.q_goal, atol=1.0e-12)


def test_policy_does_not_mutate_reference_inputs() -> None:
    solver = _make_solver(dof=3)

    def _solve(goal_xyz: np.ndarray, q_seed: np.ndarray) -> IKResult:
        del goal_xyz
        return IKResult(success=True, q=np.asarray(q_seed, dtype=float), iterations=3, residual_norm=1.0e-4)

    solver.solve = _solve  # type: ignore[method-assign]
    policy = IKPolicyConfig(
        max_ik_residual_accept_m=0.02,
        ik_random_restarts=2,
        ik_seed_default_span=np.pi,
        max_joint_jump_rad=2.0,
        use_heuristic_seed=True,
    )
    q_ref = np.array([0.10, -0.20, 0.30], dtype=float)
    q_measured = np.array([0.11, -0.19, 0.29], dtype=float)
    q_ref_before = q_ref.copy()
    q_measured_before = q_measured.copy()

    result = solver.solve_with_policy(
        goal_xyz=np.array([0.2, 0.0, 0.35], dtype=float),
        q_ref=q_ref,
        q_measured=q_measured,
        policy=policy,
        rng=np.random.default_rng(0),
    )

    assert result.reason == "accepted"
    np.testing.assert_allclose(q_ref, q_ref_before, atol=0.0)
    np.testing.assert_allclose(q_measured, q_measured_before, atol=0.0)
