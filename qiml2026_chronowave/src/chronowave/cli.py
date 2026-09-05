from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .data import load_elliptic
from .train import summarize, train_one_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent ChronoWave-GNN reproduction")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    bundle = load_elliptic(
        data_cfg["dir"],
        split=data_cfg.get("split", "paper_stratified"),
        seed=int(data_cfg.get("split_seed", 42)),
        train_max_time=int(data_cfg.get("train_max_time", 30)),
        val_max_time=int(data_cfg.get("val_max_time", 34)),
        include_timestep_in_raw_features=bool(data_cfg.get("include_timestep_in_raw_features", False)),
    )

    out_dir = Path(cfg.get("output_dir", "outputs/chronowave"))
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {k: v for k, v in bundle.metadata.items() if k not in {"raw_scaler", "wave_scaler"}}
    with open(out_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    seeds = [int(s) for s in cfg.get("seeds", [11, 22, 33, 44, 55])]
    results = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        result = train_one_seed(bundle, cfg, seed, out_dir)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    summary = summarize(results)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
