# download_nyuv2.py
"""
Step 0: scarica i file raw di NYUv2.
    python download_nyuv2.py --out_dir data/nyuv2
"""

import argparse
import os
import urllib.request


MAT_URL    = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
SPLITS_URL = "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"


def _progress(count, block, total):
    pct = min(count * block / total * 100, 100)
    bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
    print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


def download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"[skip] già presente: {dest}")
        return
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def main(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    download(MAT_URL,    os.path.join(out_dir, "nyu_depth_v2_labeled.mat"))
    download(SPLITS_URL, os.path.join(out_dir, "splits.mat"))
    print("\n[done] entrambi i file scaricati.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/nyuv2")
    main(**vars(p.parse_args()))