# Pre-computed quintic trajectory from plan_compute_node → ctrl_node.
# Arrays are ordered by motor_ids (ascending).

uint32 serial
int32 n_dof

float64[] start_q       # q snapshot at planning time (for rewarp drift)
float64[] coeffs        # flattened [n_dof × 6] leg1 coefficients
float64 duration        # leg1 duration (s)
float64[] end_q         # leg1 final positions
float64[] target_xyz    # 3 elements
float64 target_yaw

bool has_leg2
float64[] leg2_coeffs
float64 leg2_duration
float64[] leg2_end_q
float64[] leg2_target_xyz
float64 leg2_target_yaw
