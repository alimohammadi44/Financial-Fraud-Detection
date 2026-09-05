from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


BASE_URL = "https://data.pyg.org/datasets/elliptic"
FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the original Elliptic CSVs from PyTorch Geometric's public mirror"
    )
    parser.add_argument("--output", default="data/raw/elliptic")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        csv_path = out_dir / filename
        if csv_path.exists() and not args.force:
            print(f"exists: {csv_path}")
            continue

        zip_path = out_dir / f"{filename}.zip"
        url = f"{BASE_URL}/{filename}.zip"
        print(f"downloading {url}")
        download(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as archive:
            members = archive.namelist()
            if filename not in members:
                raise RuntimeError(f"Expected {filename} in {zip_path}; found {members}")
            archive.extract(filename, out_dir)
        zip_path.unlink()
        print(f"ready: {csv_path} ({csv_path.stat().st_size:,} bytes)")

    print("\nDataset downloaded. Cite Weber et al. (2019) and respect the original Elliptic CC BY-NC-ND 4.0 license.")


if __name__ == "__main__":
    main()
