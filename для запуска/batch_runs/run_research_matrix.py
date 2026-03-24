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

PACKS = {
    "pack01": "research_p01_",
    "pack02": "research_p02_",
    "pack03": "research_p03_",
    "pack04": "research_p04_",
    "pack05": "research_p05_",
    "pack06": "research_p06_",
}


def main():
    parser = argparse.ArgumentParser(description="Batch runner for research configs")
    parser.add_argument("--pack", default="all", help="all or one of: pack01..pack06")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    cfg_dir = Path(__file__).resolve().parents[1] / "configs" / "research_pack"
    configs = sorted(cfg_dir.glob("*.yaml"))
    if args.pack != "all":
        prefix = PACKS[args.pack]
        configs = [p for p in configs if p.stem.startswith(prefix)]
    print(f"Будет запущено {len(configs)} конфигов")
    for path in configs:
        print(f"\n=== Запуск {path.name} ===")
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        out = run_demo_experiment(cfg, force_regenerate=args.force_regenerate, force_retrain=args.force_retrain)
        print("Результаты сохранены в:", out)

if __name__ == "__main__":
    main()
