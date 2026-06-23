"""Self-collision checking via pinocchio + hpp-fcl.

Builds a collision geometry model from the same URDF that ``RobotModel`` uses,
applies an SRDF to exclude always-adjacent / by-design-non-colliding pairs,
and exposes per-configuration and per-trajectory collision checks.

Cage collision (arm-vs-cage mesh) is off by default and can be toggled at
runtime with ``set_cage_enabled()`` — safe to call from any thread.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pinocchio as pin

from .robot_model import RobotModel


class CollisionChecker:
    """Pinocchio-backed self-collision checker.

    Loads the URDF's collision geometry (referenced meshes), enumerates all
    pairs, then removes pairs declared in the SRDF. ``check()`` returns True
    if any remaining pair is in collision at the given configuration.

    Cage geometry objects are identified by name (``"cage"`` substring).
    Their collision pairs are included in the model but deactivated via
    ``GeometryData.activeCollisionPairs`` until ``set_cage_enabled(True)`` is
    called.
    """

    def __init__(
        self,
        robot_model: RobotModel,
        srdf_path: Optional[str | Path] = None,
        package_dirs: Optional[Iterable[str | Path]] = None,
        cage_enabled: bool = False,
    ):
        self.robot = robot_model
        self._lock = threading.Lock()

        pkg_dirs = [str(p) for p in (package_dirs or [])]
        self.geom_model = pin.buildGeomFromUrdf(
            robot_model.model,
            str(robot_model.urdf_path),
            pin.GeometryType.COLLISION,
            package_dirs=pkg_dirs,
        )
        self.geom_model.addAllCollisionPairs()
        self._n_pairs_total = len(self.geom_model.collisionPairs)

        self.srdf_path: Optional[Path] = None
        if srdf_path is not None:
            srdf = Path(srdf_path).expanduser().resolve()
            if not srdf.exists():
                raise FileNotFoundError(f"SRDF not found: {srdf}")
            pin.removeCollisionPairs(
                robot_model.model, self.geom_model, str(srdf), verbose=False
            )
            self.srdf_path = srdf

        self._n_pairs_active = len(self.geom_model.collisionPairs)

        # Cage geometry object indices (by name)
        self._cage_geom_ids: frozenset[int] = frozenset(
            i
            for i, obj in enumerate(self.geom_model.geometryObjects)
            if "cage" in obj.name.lower()
        )
        # Collision pair indices that involve at least one cage geometry object
        self._cage_pair_indices: list[int] = [
            k
            for k, pair in enumerate(self.geom_model.collisionPairs)
            if pair.first in self._cage_geom_ids or pair.second in self._cage_geom_ids
        ]

        self.geom_data = pin.GeometryData(self.geom_model)
        self._cage_enabled = False
        self.set_cage_enabled(cage_enabled)

    def set_cage_enabled(self, enabled: bool) -> None:
        """Toggle cage collision pairs on or off (thread-safe)."""
        with self._lock:
            self._cage_enabled = enabled
            for k in self._cage_pair_indices:
                self.geom_data.activeCollisionPairs[k] = enabled

    @property
    def cage_enabled(self) -> bool:
        return self._cage_enabled

    @property
    def n_pairs_total(self) -> int:
        return self._n_pairs_total

    @property
    def n_pairs_active(self) -> int:
        return self._n_pairs_active

    def check(
        self,
        q_by_motor: Mapping[int, float],
        stop_at_first: bool = True,
    ) -> bool:
        """Return True if any active pair is in collision at the configuration."""
        with self._lock:
            with self.robot._lock:
                self.robot._fill_q_buf(q_by_motor)
                return bool(
                    pin.computeCollisions(
                        self.robot.model,
                        self.robot.data,
                        self.geom_model,
                        self.geom_data,
                        self.robot._q_buf,
                        stop_at_first_collision=stop_at_first,
                    )
                )

    def colliding_pairs(
        self, q_by_motor: Mapping[int, float]
    ) -> list[tuple[str, str]]:
        """List of colliding ``(link_a, link_b)`` link-name pairs at the configuration.

        Computes all pairs (no early-exit), so slower than ``check()``.
        """
        self.check(q_by_motor, stop_at_first=False)
        out: list[tuple[str, str]] = []
        for k, result in enumerate(self.geom_data.collisionResults):
            if result.isCollision():
                pair = self.geom_model.collisionPairs[k]
                obj_a = self.geom_model.geometryObjects[pair.first]
                obj_b = self.geom_model.geometryObjects[pair.second]
                out.append((obj_a.name, obj_b.name))
        return out

    def check_trajectory(
        self,
        q_samples: Iterable[Mapping[int, float]],
    ) -> tuple[bool, int]:
        """Check a sequence of configurations.

        Returns ``(any_collision, first_collision_index)``. ``first_collision_index``
        is -1 if no collision was found.
        """
        for idx, q in enumerate(q_samples):
            if self.check(q, stop_at_first=True):
                return True, idx
        return False, -1
