"""모터 파라미터 정의/조회 테이블."""

# lib/params.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ParamDef:
    index: int
    dtype: str      # "float" | "uint8" | "uint16" | "uint32"
    access: str     # "R" | "W/R" | "W"

# RS03 single parameter list (0x7005~0x702B) + aliases
PARAMS: dict[str, ParamDef] = {
    "run_mode":      ParamDef(0x7005, "uint8",  "W/R"),
    "iq_ref":        ParamDef(0x7006, "float",  "W/R"),
    "spd_ref":       ParamDef(0x700A, "float",  "W/R"),
    "limit_torque":  ParamDef(0x700B, "float",  "W/R"),

    "cur_kp":        ParamDef(0x7010, "float",  "W/R"),
    "cur_ki":        ParamDef(0x7011, "float",  "W/R"),
    "cur_filt_gain": ParamDef(0x7014, "float",  "W/R"),

    "loc_ref":       ParamDef(0x7016, "float",  "W/R"),
    "limit_spd":     ParamDef(0x7017, "float",  "W/R"),
    "limit_cur":     ParamDef(0x7018, "float",  "W/R"),

    "mechPos":       ParamDef(0x7019, "float",  "R"),
    "iqf":           ParamDef(0x701A, "float",  "R"),
    "mechVel":       ParamDef(0x701B, "float",  "R"),
    "VBUS":          ParamDef(0x701C, "float",  "R"),

    "loc_kp":        ParamDef(0x701E, "float",  "W/R"),
    "spd_kp":        ParamDef(0x701F, "float",  "W/R"),
    "spd_ki":        ParamDef(0x7020, "float",  "W/R"),
    "spd_filt_gain": ParamDef(0x7021, "float",  "W/R"),

    "acc_rad":       ParamDef(0x7022, "float",  "W/R"),
    "vel_max":       ParamDef(0x7024, "float",  "W/R"),
    "acc_set":       ParamDef(0x7025, "float",  "W/R"),

    "EPScan_time":   ParamDef(0x7026, "uint16", "W"),
    "canTimeout":    ParamDef(0x7028, "uint32", "W"),
    "zero_sta":      ParamDef(0x7029, "uint8",  "W"),
    "damper":        ParamDef(0x702A, "uint8",  "W/R"),
    "add_offset":    ParamDef(0x702B, "float",  "W/R"),
    "alveolous_open": ParamDef(0x702C, "uint8", "W/R"),  # RS05 cogging compensation switch
    "iq_test":       ParamDef(0x702D, "uint8",  "W/R"),  # RS05 motor initialization calibration
    "dcc_set":       ParamDef(0x702E, "float",  "W/R"),  # RS05 PP-mode deceleration

    # aliases
    "q_vel":         ParamDef(0x701B, "float",  "R"),
    "q_pos":         ParamDef(0x7019, "float",  "R"),
    "bus_v":         ParamDef(0x701C, "float",  "R"),
}

def resolve_param(name_or_index: str) -> ParamDef:
    s = name_or_index.strip()
    if s in PARAMS:
        return PARAMS[s]
    # allow 0x701E
    idx = int(s, 0)
    # 타입 모르면 raw로 확인하는 용도
    return ParamDef(idx, "uint32", "R")

def iter_params():
    return sorted(PARAMS.items(), key=lambda kv: kv[0])
