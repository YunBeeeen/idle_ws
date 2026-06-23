"""튜닝 파라미터를 저장한다."""

import argparse

from lib.control_tuning import save_control_tuning


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    path = save_control_tuning()
    print(f"saved control tuning: tuned_file={path}")


if __name__ == "__main__":
    main()
