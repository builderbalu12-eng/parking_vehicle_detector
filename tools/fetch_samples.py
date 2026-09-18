#!/usr/bin/env python3
"""Download the sample image and video used by the README examples.

Everything fetched here is openly licensed; attribution is written to
data/input/SOURCES.md so the credit travels with the files.

Usage:
    python tools/fetch_samples.py            # the two files the README uses
    python tools/fetch_samples.py --all      # plus an extra mixed-class video
    python tools/fetch_samples.py --force    # re-download even if present
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"

# Wikimedia rejects requests without a normal browser User-Agent, and it also
# blocks its /thumb/ paths, so we pull the original and downscale locally.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

SAMPLES: list[dict] = [
    {
        "name": "parking_lot.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Parking_Lot_with_Cars_in_Cape_Town_CBD.jpg",
        "what": "Elevated view of a busy CBD parking lot, plus a public street on the right. "
                "The street is what makes the ROI bonus worth demonstrating.",
        "author": "Husskeyy",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
        "page": "https://commons.wikimedia.org/wiki/File:Parking_Lot_with_Cars_in_Cape_Town_CBD.jpg",
        "max_width": 2048,   # downscale: the original is 4096 px and ~6 MB
        "optional": False,
    },
    {
        "name": "street_traffic.mp4",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti-inference-vid.mp4",
        "what": "Street-level drive through Karlsruhe with moving and parked cars. "
                "Ground-level framing, so the tracker has something to follow.",
        "author": "KITTI Vision Benchmark Suite (Geiger et al.), via ultralytics/assets",
        "license": "CC BY-NC-SA 3.0 (non-commercial)",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
        "page": "https://www.cvlibs.net/datasets/kitti/",
        "max_width": None,
        "optional": False,
    },
    {
        "name": "topdown_lot.mp4",
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
        "what": "FAILURE CASE, kept on purpose. A near-vertical bird's-eye view of a "
                "parking lot on which COCO-trained YOLO detects almost nothing. "
                "See the limitations section of NOTES.md.",
        "author": "Intel IoT DevKit",
        "license": "See repository terms",
        "license_url": "https://github.com/intel-iot-devkit/sample-videos",
        "page": "https://github.com/intel-iot-devkit/sample-videos",
        "max_width": None,
        "optional": True,
    },
]


def download(url: str, dest: Path) -> None:
    """Fetch `url` to `dest`, showing a simple progress line."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            while chunk := response.read(1 << 16):
                fh.write(chunk)
                read += len(chunk)
                if total:
                    pct = 100 * read / total
                    print(f"\r    {read/1e6:5.1f} / {total/1e6:.1f} MB ({pct:3.0f}%)",
                          end="", flush=True)
        print()
        tmp.replace(dest)


def downscale(path: Path, max_width: int) -> None:
    """Shrink an image in place if it is wider than `max_width`."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        print(f"    ! could not read {path.name} to downscale")
        return
    h, w = img.shape[:2]
    if w <= max_width:
        return
    scale = max_width / w
    resized = cv2.resize(img, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"    downscaled {w}x{h} -> {resized.shape[1]}x{resized.shape[0]}")


def write_sources(samples: list[dict]) -> Path:
    """Record where each sample came from, so credit ships with the repo."""
    lines = [
        "# Sample media sources",
        "",
        "These files are downloaded by `python tools/fetch_samples.py` and are not",
        "committed to the repository.",
        "",
    ]
    for s in samples:
        lines += [
            f"## {s['name']}",
            "",
            f"- {s['what']}",
            f"- Author: {s['author']}",
            f"- License: [{s['license']}]({s['license_url']})",
            f"- Source: {s['page']}",
            "",
        ]
    path = INPUT_DIR / "SOURCES.md"
    path.write_text("\n".join(lines))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="also fetch the optional top-down failure-case clip")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [s for s in SAMPLES if args.all or not s["optional"]]

    failures = 0
    for sample in wanted:
        dest = INPUT_DIR / sample["name"]
        if dest.exists() and not args.force:
            print(f"[skip] {sample['name']} already present ({dest.stat().st_size/1e6:.1f} MB)")
            continue

        print(f"[get ] {sample['name']}")
        try:
            download(sample["url"], dest)
            if sample["max_width"]:
                downscale(dest, sample["max_width"])
        except Exception as exc:
            failures += 1
            print(f"    ! failed: {exc}")
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)

    sources = write_sources(wanted)
    print(f"\nAttribution written to {sources.relative_to(ROOT)}")
    print(f"Samples live in {INPUT_DIR.relative_to(ROOT)}/")

    if failures:
        print(f"\n{failures} download(s) failed. Check your network and retry.")
        return 1

    print("\nNext:")
    print("  python detect_vehicles.py --source data/input/parking_lot.jpg")
    print("  python detect_vehicles.py --source data/input/car-detection.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
