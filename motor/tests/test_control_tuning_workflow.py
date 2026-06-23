"""control_tuning 수정/저장 워크플로 검증 테스트."""

from pathlib import Path
import sys
import os
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.control_tuning import (
    control_session,
    save_control_tuning,
    set_control_tuning,
)
from lib.param_store import control_tuned_path, load_control_tuned


class ControlTuningWorkflowTest(unittest.TestCase):
    def test_modify_save_then_readback(self):
        old_param_root = os.environ.get("IDLE_PARAM_ROOT")
        old_state_path = os.environ.get("IDLE_CONTROL_GATE_STATE")
        try:
            with tempfile.TemporaryDirectory() as td:
                param_root = Path(td) / "param"
                state_file = Path(td) / "control_state.json"
                os.environ["IDLE_PARAM_ROOT"] = str(param_root)
                os.environ["IDLE_CONTROL_GATE_STATE"] = str(state_file)

                set_control_tuning([1], {"kp": 12.5, "kd": 0.2})
                save_control_tuning()

                tuned = load_control_tuned()
                self.assertIn("motors", tuned)
                self.assertAlmostEqual(float(tuned["motors"]["1"]["kp"]), 12.5)
                self.assertAlmostEqual(float(tuned["motors"]["1"]["kd"]), 0.2)
                self.assertTrue(control_tuned_path().exists())
                self.assertFalse(state_file.exists())
        finally:
            if old_param_root is None:
                os.environ.pop("IDLE_PARAM_ROOT", None)
            else:
                os.environ["IDLE_PARAM_ROOT"] = old_param_root
            if old_state_path is None:
                os.environ.pop("IDLE_CONTROL_GATE_STATE", None)
            else:
                os.environ["IDLE_CONTROL_GATE_STATE"] = old_state_path

    def test_tx_hz_key_is_no_longer_supported(self):
        old_param_root = os.environ.get("IDLE_PARAM_ROOT")
        old_state_path = os.environ.get("IDLE_CONTROL_GATE_STATE")
        try:
            with tempfile.TemporaryDirectory() as td:
                param_root = Path(td) / "param"
                state_file = Path(td) / "control_state.json"
                os.environ["IDLE_PARAM_ROOT"] = str(param_root)
                os.environ["IDLE_CONTROL_GATE_STATE"] = str(state_file)

                with self.assertRaises(ValueError):
                    set_control_tuning([3], {"tx_hz": 180.0})
        finally:
            if old_param_root is None:
                os.environ.pop("IDLE_PARAM_ROOT", None)
            else:
                os.environ["IDLE_PARAM_ROOT"] = old_param_root
            if old_state_path is None:
                os.environ.pop("IDLE_CONTROL_GATE_STATE", None)
            else:
                os.environ["IDLE_CONTROL_GATE_STATE"] = old_state_path

    def test_control_session_blocks_parameter_edit(self):
        old_param_root = os.environ.get("IDLE_PARAM_ROOT")
        old_state_path = os.environ.get("IDLE_CONTROL_GATE_STATE")
        try:
            with tempfile.TemporaryDirectory() as td:
                param_root = Path(td) / "param"
                state_file = Path(td) / "control_state.json"
                os.environ["IDLE_PARAM_ROOT"] = str(param_root)
                os.environ["IDLE_CONTROL_GATE_STATE"] = str(state_file)

                with control_session():
                    with self.assertRaises(RuntimeError):
                        set_control_tuning([1], {"kp": 5.0})
        finally:
            if old_param_root is None:
                os.environ.pop("IDLE_PARAM_ROOT", None)
            else:
                os.environ["IDLE_PARAM_ROOT"] = old_param_root
            if old_state_path is None:
                os.environ.pop("IDLE_CONTROL_GATE_STATE", None)
            else:
                os.environ["IDLE_CONTROL_GATE_STATE"] = old_state_path


if __name__ == "__main__":
    unittest.main()
