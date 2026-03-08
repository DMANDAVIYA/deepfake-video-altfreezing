"""
run.py — CLI entry point for AltFreezing batch inference.
All model/inference logic lives in detector.py.

Usage:
    python pipeline/run.py
    python pipeline/run.py --input D:/videos --ckpt D:/model.pth
"""
import sys, os, csv, argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
_PIPELINE_DIR  = Path(__file__).resolve().parent
_ALTFREEZING   = (_PIPELINE_DIR.parent / "AltFreezing").resolve()
_DEFAULT_INPUT  = _PIPELINE_DIR / "data" / "input"
_DEFAULT_OUTPUT = _PIPELINE_DIR / "data" / "output"
_DEFAULT_CKPT   = _ALTFREEZING / "checkpoints" / "model.pth"

assert _ALTFREEZING.exists(), (
    f"AltFreezing not found at {_ALTFREEZING}\n"
    "Clone it: git clone https://github.com/ZhendongWang6/AltFreezing.git"
)

# must happen BEFORE importing detector.py (face weights download to ./auxillary/)
os.chdir(str(_ALTFREEZING))
sys.path.insert(0, str(_ALTFREEZING))
sys.path.insert(0, str(_PIPELINE_DIR))

from detector import load_model, infer_video, THRESHOLD


def parse_args():
    p = argparse.ArgumentParser(description="AltFreezing deepfake batch detection")
    p.add_argument("--input",  default=str(_DEFAULT_INPUT),  help="folder of .mp4 files")
    p.add_argument("--output", default=str(_DEFAULT_OUTPUT), help="output folder for CSV")
    p.add_argument("--ckpt",   default=str(_DEFAULT_CKPT),   help="path to model.pth")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt)
    inp  = Path(args.input)
    out  = Path(args.output)

    assert ckpt.exists(), (
        f"model.pth not found: {ckpt}\n"
        "Download from https://rec.ustc.edu.cn/share/e87360b0-7b2e-11ef-aeef-a9fd0832d537"
        " (password: altf) and place in AltFreezing/checkpoints/"
    )

    videos = sorted(inp.glob("*.mp4"))
    assert videos, f"No .mp4 files in {inp}"
    print(f"{len(videos)} video(s) | ckpt: {ckpt.name} | threshold: {THRESHOLD}")

    print("Loading model...")
    classifier, crop_align_func = load_model(str(ckpt))
    print("Ready\n")

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "score", "prediction"])

        for video in tqdm(videos, desc="Inferring"):
            try:
                score = infer_video(str(video), classifier, crop_align_func)
            except Exception as e:
                tqdm.write(f"  ERROR {video.name}: {e}")
                score = -1.0

            prediction = "FAKE" if score >= THRESHOLD else "real"
            writer.writerow([video.name, f"{score:.4f}", prediction])
            f.flush()
            tqdm.write(f"  {video.name:<55} {prediction:<6}  ({score:.4f})")

    print(f"\nResults → {csv_path}")


if __name__ == "__main__":
    main()
