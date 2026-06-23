#!/usr/bin/env python3
"""Regenerate src/sim/robot.xml from src/sim/urdf/robot.urdf using MuJoCo's URDF parser.

Run this whenever robot.urdf changes (new joints, geometry updates, etc.)
so that MuJoCo-using nodes (viewer_node, sim_driver_node) stay in sync with
the URDF used by Pinocchio-based modules (robot_model, IK, collision).

Usage:
    python3 src/sim/scripts/regen_robot_xml.py

Effects:
    - Writes src/sim/robot.xml
    - Uses relative meshdir='meshes/' for portability
    - Preserves URDF source untouched
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path


def main() -> int:
    try:
        import mujoco
    except ImportError:
        print("ERROR: mujoco module not installed", file=sys.stderr)
        return 1

    sim_pkg = Path(__file__).resolve().parents[1]
    urdf_src = sim_pkg / "urdf" / "robot.urdf"
    xml_dst = sim_pkg / "robot.xml"

    if not urdf_src.exists():
        print(f"ERROR: URDF not found: {urdf_src}", file=sys.stderr)
        return 1

    install_meshdir = Path("/home/su/idle_ws/install/sim/share/sim/meshes/")
    if not install_meshdir.exists():
        print(
            f"WARNING: install meshdir not found at {install_meshdir}\n"
            "Run 'colcon build --packages-select sim' first so meshes are available.",
            file=sys.stderr,
        )
        return 1

    text = urdf_src.read_text()
    hint = (
        f'<mujoco>'
        f'<compiler meshdir="{install_meshdir}" balanceinertia="true" '
        f'strippath="true" discardvisual="false"/>'
        f'</mujoco>'
    )
    text_with_hint = text.replace(
        '<robot name="my_robot">',
        f'<robot name="my_robot">\n  {hint}',
        1,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", delete=False
    ) as tmp:
        tmp.write(text_with_hint)
        tmp_path = tmp.name

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
        mujoco.mj_saveLastXML(str(xml_dst), model)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Rewrite the absolute meshdir as relative for portability
    xml_text = xml_dst.read_text()
    xml_text = re.sub(r'meshdir="[^"]*"', 'meshdir="meshes/"', xml_text)
    xml_dst.write_text(xml_text)

    print(f"Regenerated: {xml_dst}")
    print(f"  joints: {model.njnt}")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jt = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        print(f"    [{i}] {name} ({jt})")
    print("Reminder: rebuild sim package — colcon build --packages-select sim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
