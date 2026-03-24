from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / 'src'
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from для_показа.demo_helpers import run_demo_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run one research/demo config")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = run_demo_experiment(cfg, force_regenerate=args.force_regenerate, force_retrain=args.force_retrain)
    print("Результаты сохранены в:", out)


if __name__ == "__main__":
    main()
