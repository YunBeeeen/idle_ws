"""현재 유효 제어 튜닝값을 출력."""

import argparse
import json

from lib.control_tuning import control_params_for_motor
from lib.param_store import control_tuned_path, load_control_effective


def main():
    ap = argparse.ArgumentParser(description="Show effective control tuning params.")
    ap.add_argument("--can_id", type=lambda x: int(x, 0), nargs="+", default=None)
    args = ap.parse_args()

    if args.can_id:
        for can_id in args.can_id:
            values = control_params_for_motor(can_id)
            print(f"can_id={can_id} tuning={json.dumps(values, sort_keys=True)}")
    else:
        doc = load_control_effective()
        print(json.dumps(doc, indent=2, sort_keys=True))

    print(f"tuned_file={control_tuned_path()}")


if __name__ == "__main__":
    main()
