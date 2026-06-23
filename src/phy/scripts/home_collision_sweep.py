#!/usr/bin/env python3
"""Find the j3 (elbow) angle window in which j2 (shoulder) can reach home
without self-collision.

Background: returning j2 to home is only safe when j3 is beyond a certain
angle. This script sweeps j3 (and optionally the full j2x j3 grid) through the
CollisionChecker so the home-return sequencer knows the safe corridor instead
of relying on can_bridge's blind per-joint interpolation.

Runs directly against the source tree (no colcon build needed) so it picks up
the freshly edited SRDF. Requires pinocchio + hpp-fcl (coal).

Examples:
    # Safe j3 window with j2 at home, and the boundary as j2 varies:
    python3 src/phy/scripts/home_collision_sweep.py

    # Sanity check a single pose (degrees), list colliding pairs:
    python3 src/phy/scripts/home_collision_sweep.py --pose 0 0 0 0 0 0

    # Reprint URDF adjacency for the SRDF:
    python3 src/phy/scripts/home_collision_sweep.py --dump-adjacent
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_PHY_PKG = _SCRIPTS.parent          # src/phy   (contains the `phy` package)
_SRC = _PHY_PKG.parent              # src
sys.path.insert(0, str(_PHY_PKG))
sys.path.insert(0, str(_SRC / "idle_common"))

from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP  # noqa: E402
from phy.collision import CollisionChecker  # noqa: E402
from phy.robot_model import RobotModel  # noqa: E402

DEFAULT_URDF = _SRC / "sim" / "urdf" / "robot.urdf"
DEFAULT_SRDF = _SRC / "sim" / "srdf" / "robot.srdf"
DEFAULT_PKG = _SRC                  # so package://sim/meshes/... resolves

# Joints not being swept are held at this q during the sweep (their home).
HOME_BY_MOTOR = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
SHOULDER = 2  # j2
ELBOW = 3     # j3


def _dump_adjacent() -> int:
    import xml.etree.ElementTree as ET

    root = ET.parse(DEFAULT_URDF).getroot()
    print("<!-- Adjacent kinematic chain (parent -> child in URDF tree) -->")
    for j in root.findall("joint"):
        p = j.find("parent").get("link")
        c = j.find("child").get("link")
        print(f'  <disable_collisions link1="{p}" link2="{c}" reason="Adjacent"/>')
    return 0


def _build(args) -> tuple[RobotModel, CollisionChecker]:
    rm = RobotModel(str(args.urdf), dict(DEFAULT_MOTOR_JOINT_MAP))
    cc = CollisionChecker(rm, srdf_path=str(args.srdf), package_dirs=[str(args.pkg)])
    print(
        f"collision model: {cc.n_pairs_active} active pairs "
        f"({cc.n_pairs_total} total, {cc.n_pairs_total - cc.n_pairs_active} excluded by SRDF)"
    )
    return rm, cc


def _q(overrides: dict[int, float]) -> dict[int, float]:
    q = dict(HOME_BY_MOTOR)
    q.update(overrides)
    return q


def _report_pairs(cc: CollisionChecker, q: dict[int, float]) -> None:
    for a, b in cc.colliding_pairs(q):
        print(f"      collide: {a} <-> {b}")


def _intervals(angles: list[float], free: list[bool]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    start = None
    for ang, ok in zip(angles, free):
        if ok and start is None:
            start = ang
        elif not ok and start is not None:
            out.append((start, prev))
            start = None
        prev = ang
    if start is not None:
        out.append((start, angles[-1]))
    return out


def _sweep_elbow(cc: CollisionChecker, j2: float, lo: float, hi: float, step: float):
    angles = list(np.arange(lo, hi + 1e-9, step))
    free = [not cc.check(_q({SHOULDER: j2, ELBOW: a}), stop_at_first=True) for a in angles]
    return angles, free


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--srdf", type=Path, default=DEFAULT_SRDF)
    ap.add_argument("--pkg", type=Path, default=DEFAULT_PKG)
    ap.add_argument("--step-deg", type=float, default=2.0, help="j3 sweep resolution")
    ap.add_argument("--j2-grid-deg", type=float, default=15.0, help="j2 step for the 2D boundary table")
    ap.add_argument("--pose", type=float, nargs=6, metavar="DEG",
                    help="check one pose j1..j6 in degrees, list colliding pairs, exit")
    ap.add_argument("--dump-adjacent", action="store_true", help="print URDF adjacency for SRDF and exit")
    args = ap.parse_args()

    if args.dump_adjacent:
        return _dump_adjacent()

    rm, cc = _build(args)
    lower, upper = rm.joint_limits()
    idx = {mid: i for i, mid in enumerate(rm.ordered_motor_ids)}
    j2_lo, j2_hi = float(lower[idx[SHOULDER]]), float(upper[idx[SHOULDER]])
    j3_lo, j3_hi = float(lower[idx[ELBOW]]), float(upper[idx[ELBOW]])
    if not np.isfinite([j3_lo, j3_hi]).all():
        j3_lo, j3_hi = -math.pi, math.pi
    if not np.isfinite([j2_lo, j2_hi]).all():
        j2_lo, j2_hi = -math.pi, math.pi

    rad = math.radians

    if args.pose is not None:
        q = {m: rad(d) for m, d in zip(range(1, 7), args.pose)}
        hit = cc.check(q, stop_at_first=False)
        print(f"pose {args.pose} deg -> {'COLLISION' if hit else 'clear'}")
        if hit:
            _report_pairs(cc, q)
        return 0

    print(f"\nlimits: j2 in [{math.degrees(j2_lo):.0f}, {math.degrees(j2_hi):.0f}] deg, "
          f"j3 in [{math.degrees(j3_lo):.0f}, {math.degrees(j3_hi):.0f}] deg")
    print("(non-swept joints j1,j4,j5,j6 held at 0; fingers closed)\n")

    # 1) j2 at home: safe j3 window
    print("=== j2 = home (0 deg): collision-free j3 window ===")
    angles, free = _sweep_elbow(cc, 0.0, j3_lo, j3_hi, rad(args.step_deg))
    free_iv = _intervals(angles, free)
    if not free_iv:
        print("  NONE — j2=home collides at every j3 in range (check SRDF false positives below)")
    for a, b in free_iv:
        print(f"  safe j3 in [{math.degrees(a):+.0f}, {math.degrees(b):+.0f}] deg")
    # show pairs at a colliding sample (if any) to spot SRDF false positives
    coll = [a for a, ok in zip(angles, free) if not ok]
    if coll:
        s = coll[len(coll) // 2]
        print(f"  example collision at j2=0, j3={math.degrees(s):+.0f} deg:")
        _report_pairs(cc, _q({SHOULDER: 0.0, ELBOW: s}))

    # 2) boundary as j2 varies: min safe j3 (the "j3 must be beyond X" rule)
    print("\n=== safe j3 window vs j2 (the home-approach corridor) ===")
    print("   j2[deg]   safe j3 window(s) [deg]")
    j2s = list(np.arange(j2_lo, j2_hi + 1e-9, rad(args.j2_grid_deg)))
    for j2 in j2s:
        angles, free = _sweep_elbow(cc, j2, j3_lo, j3_hi, rad(args.step_deg))
        iv = _intervals(angles, free)
        txt = ", ".join(f"[{math.degrees(a):+.0f}, {math.degrees(b):+.0f}]" for a, b in iv) or "NONE"
        print(f"   {math.degrees(j2):+6.0f}    {txt}")

    print("\nNote: angles are in the CURRENT URDF j3 convention. After the j3 hardware")
    print("re-zero + URDF origin update, re-run this — the physical safe corridor is")
    print("unchanged but the numeric j3 values shift by the re-zero offset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
