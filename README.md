# Deepfake Detection Pipeline — AltFreezing

Batch deepfake detection using [AltFreezing](https://github.com/ZhendongWang6/AltFreezing) (CVPR 2023).  
Runs on GPU (CUDA required). Processes a folder of `.mp4` videos and outputs a CSV with labels and scores.

---

## How It Works

AltFreezing uses an I3D spatiotemporal network that analyzes 32-frame sliding window clips.  
Each video gets a **score between 0.0 and 1.0**:
- Score ≥ 0.5 → **FAKE**
- Score < 0.5 → **real**

---

## Project Structure

```
pipeline/
├── detector.py       # model loading + per-video inference
├── run.py            # CLI batch runner
├── run.ipynb         # Google Colab notebook
├── requirements.txt  # dependencies
└── data/
    ├── input/        # put your .mp4 videos here (local use)
    └── output/       # results CSV written here
AltFreezing/          # cloned model repo (see setup)
```

---

## Setup (Local — GPU Required)

### 1. Clone AltFreezing

```bash
git clone https://github.com/ZhendongWang6/AltFreezing.git
```

### 2. Download model weights

Download `model.pth` from the [AltFreezing release](https://rec.ustc.edu.cn/share/e87360b0-7b2e-11ef-aeef-a9fd0832d537) and place it at:

```
AltFreezing/checkpoints/model.pth
```

### 3. Install dependencies

```bash
pip install -r pipeline/requirements.txt
```

> Edit `requirements.txt` first — change `cu118` to match your CUDA version (`cu118` / `cu121` / `cu124`).  
> Check your version with: `nvidia-smi`

---

## Usage

### Option A — CLI (local)

Put your `.mp4` files in `pipeline/data/input/`, then run:

```bash
python pipeline/run.py
```

**Custom paths:**

```bash
python pipeline/run.py --input D:/videos --ckpt D:/model.pth --output D:/results
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--input` | `pipeline/data/input` | Folder containing `.mp4` files |
| `--output` | `pipeline/data/output` | Folder where CSV is saved |
| `--ckpt` | `AltFreezing/checkpoints/model.pth` | Path to `model.pth` |

### Output CSV format

```
filename,label,score
video1.mp4,real,0.1234
video2.mp4,FAKE,0.8765
```

---

### Option B — Google Colab (recommended)

Open `pipeline/run.ipynb` in Google Colab with a **T4 GPU runtime**.

**Cell-by-cell:**

| Cell | What it does |
|---|---|
| 1 | Checks GPU is available |
| 2 | Clones AltFreezing + installs dependencies |
| 3 | Mounts Google Drive |
| 4 | **CONFIG** — set your paths and threshold here |
| 5 | Sets working directory + Python path |
| 6 | Writes `detector.py` to `/content/` |
| 7 | Loads the model from your checkpoint |
| 8 | Runs batch inference, prints per-video results |
| 9 | Downloads the results CSV |

**Key config in Cell 4:**

```python
CKPT_PATH  = '/content/drive/MyDrive/altfreezing/model.pth'
INPUT_DIR  = '/content/drive/MyDrive/your_videos/'
OUTPUT_DIR = '/content/results'
THRESHOLD  = 0.5  
```

> **Important:** Keep `THRESHOLD = 0.5`. Lower values (like 0.04) will label everything as FAKE.

---

## Notes

- Face detection (RetinaFace) runs on CPU — this is the main bottleneck, not the GPU.
- Expect ~10–60s per video on T4 depending on video length.
- The CSV is flushed after every video — safe to stop mid-run and download partial results.
- Input folder should contain only `.mp4` files. Subfolders are not scanned.
- If no face is detected in a video, it gets score `0.5` (neutral) and is skipped safely.

---

## Model Info

| Property | Value |
|---|---|
| Paper | AltFreezing (CVPR 2023) |
| Architecture | I3D spatiotemporal network |
| Trained on | FaceForensics++, Celeb-DF, DFDC |
| Input | Video (face required) |
| GPU required | Yes (CUDA) |
| Threshold | 0.5 (sigmoid output) |
