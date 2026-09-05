from __future__ import annotations

import argparse
import json

import yaml

from .data import load_elliptic_temporal
from .train import train_reproduction


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent FG-EGCN reproduction")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    dataset = load_elliptic_temporal(
        data_cfg["dir"],
        local_feature_dim=int(data_cfg.get("local_feature_dim", 94)),
    )
    print(json.dumps(dataset.metadata, indent=2), flush=True)

    result = train_reproduction(dataset, cfg)
    print("\n=== frozen C1 result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
