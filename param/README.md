# Param Layout

## Root resolution
- `IDLE_PARAM_ROOT` environment variable is used first.
- If unset, tools auto-detect workspace `param/`.

## Files
- `original/driver_params.yaml`: low-frequency Type18 driver params.
- `original/control_params.yaml`: baseline control tuning params.
- `tuned/control_params.yaml`: writable control tuning overrides.

## Rules
- Driver params are edited manually and applied only via `motor/apply_driver_params.py`.
- Control tuning follows `modify -> save -> control`.
- `tuned/control_params.yaml` is the only save target for control tuning.
- `can_bridge_node` 설정 변경은 코드 상수 수정 후 재빌드/재시작으로 반영한다.
