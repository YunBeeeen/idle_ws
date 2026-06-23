"""Run the contact-detection training pipeline with one explicit config.

여러 스크립트를 사람이 직접 순서대로 치지 않아도 되도록 묶은 helper다.
논문 본류 순서는 generate -> diagnose -> train -> evaluate이다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_BY_STEP = {
    "generate": "generate_sim_dataset.py",
    "diagnose": "diagnose_dataset.py",
    "train": "train_detectors.py",
    "evaluate": "evaluate_detectors.py",
    "visualize": "visualize_sim_detection.py",
}


def run_step(step: str, config_path: Path, stage: str) -> None:
    """Execute one pipeline script as a subprocess and stop immediately on failure."""
    script_dir = Path(__file__).resolve().parent
    script = script_dir / SCRIPT_BY_STEP[step]
    command = [sys.executable, str(script), "--config", str(config_path), "--stage", str(stage)]
    if step == "visualize":
        command.extend(["--episode", "median"])
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Config YAML path.")
    parser.add_argument("--stage", default="easy_hold", help="Curriculum stage to run.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(SCRIPT_BY_STEP),
        default=["generate", "diagnose", "train", "evaluate"],
        help="Pipeline steps to run in order.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Using config: {config_path}")
    print(f"Using stage: {args.stage}")
    # 각 step은 별도 Python process로 실행된다. 이렇게 하면 중간 실패 위치가 명확하고,
    # generate/train/evaluate 각각을 독립적으로 다시 실행하기 쉽다.
    for step in args.steps:
        run_step(step, config_path, args.stage)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
