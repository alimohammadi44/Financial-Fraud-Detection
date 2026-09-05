from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGETS = {
    "illicit_precision": 0.834,
    "illicit_recall": 0.722,
    "illicit_f1": 0.774,
    "micro_f1": 0.971,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FG-EGCN reproduction to Han et al. (2026)")
    parser.add_argument("--result", required=True, help="Path to result.json produced by fg_egcn.cli")
    args = parser.parse_args()

    path = Path(args.result)
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)
    observed = result["test"]

    rows = []
    for metric, published in TARGETS.items():
        value = float(observed[metric])
        rows.append(
            {
                "metric": metric,
                "published": published,
                "reproduction": value,
                "absolute_delta": value - published,
                "relative_delta_percent": 100.0 * (value - published) / published,
            }
        )

    report = {
        "paper": "Han et al., Scientific Reports 16, 22352 (2026)",
        "doi": "10.1038/s41598-026-53783-y",
        "split_used": result.get("split"),
        "seed": result.get("seed"),
        "comparison": rows,
        "note": (
            "A close numerical match is evidence of reproduction only under the documented "
            "implementation assumptions; it does not prove identity with the authors' unpublished code."
        ),
    }

    out = path.with_name("comparison_to_paper.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"{'metric':20s} {'paper':>10s} {'repro':>10s} {'delta':>10s}")
    print("-" * 55)
    for row in rows:
        print(
            f"{row['metric']:20s} {row['published']:10.4f} "
            f"{row['reproduction']:10.4f} {row['absolute_delta']:+10.4f}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
