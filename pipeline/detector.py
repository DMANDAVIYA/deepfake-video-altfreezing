"""
detector.py — AltFreezing model loading and per-video inference.
Import this from run.py or a notebook.

REQUIREMENT: caller must os.chdir(ALTFREEZING_DIR) and
             sys.path.insert(0, ALTFREEZING_DIR) before importing this module.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import config as cfg
from test_tools.common import detect_all
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader

THRESHOLD = 0.04   # optimal threshold from AltFreezing demo.py
_MAX_FRAME = 400
_MEAN = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1, 1)
_STD  = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1, 1)


def load_model(ckpt_path: str):
    """Load AltFreezing i3d_ori classifier. Returns (classifier, crop_align_func)."""
    cfg.init_with_yaml()
    cfg.update_with_yaml("i3d_ori.yaml")
    cfg.freeze()

    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier.cuda()
    classifier.eval()
    classifier.load(ckpt_path)

    crop_align_func = FasterCropAlignXRay(cfg.imsize)
    return classifier, crop_align_func


def infer_video(video_path: str, classifier, crop_align_func) -> float:
    """
    Run AltFreezing on a single video.
    Returns mean sigmoid score across all clips.
    Score >= THRESHOLD → FAKE.
    Returns 0.5 if no face is detected.
    """
    detect_res, all_lm68, frames = detect_all(
        video_path, return_frames=True, max_size=_MAX_FRAME
    )
    if not frames:
        return 0.5

    shape = frames[0].shape[:2]

    # merge 5-pt + 68-pt landmarks into unified detect_res
    merged = []
    for faces, lm68s in zip(detect_res, all_lm68):
        merged.append([
            (box, lm5, lm68, score)
            for (box, lm5, score), lm68 in zip(faces, lm68s)
        ])
    detect_res = merged

    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)
    if not tracks:
        tuples, tracks = find_longest(detect_res)
    if not tracks:
        return 0.5

    # crop faces and store per-frame landmark info
    data_storage = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        super_clips.append(len(track))
        for j, (face, frame_idx) in enumerate(zip(track, range(start, end))):
            box, lm5, lm68 = face[:3]
            big_box  = get_crop_box(shape, box, scale=0.5)
            top_left = big_box[:2][None, :]
            info = (
                (box.reshape(2, 2) - top_left).reshape(-1),
                lm5  - top_left,
                lm68 - top_left,
                big_box,
            )
            x1, y1, x2, y2 = big_box
            data_storage[f"{track_i}_{j}_img"] = frames[frame_idx][y1:y2, x1:x2]
            data_storage[f"{track_i}_{j}_ldm"] = info
            data_storage[f"{track_i}_{j}_idx"] = frame_idx

    # build sliding-window clips with palindrome padding if track is too short
    clip_size  = cfg.clip_size
    pad_length = clip_size - 1
    clips = []

    for sc_idx, sc_size in enumerate(super_clips):
        inner = list(range(sc_size))
        if sc_size < clip_size:
            post = inner[1:-1][::-1] + inner
            post = (post * (pad_length // len(post) + 1))[:pad_length]
            pre  = inner + inner[1:-1][::-1]
            pre  = (pre  * (pad_length // len(pre)  + 1))[-pad_length:]
            inner = pre + inner + post
        sc_size = len(inner)
        for i in range(sc_size):
            if i + clip_size <= sc_size:
                clips.append([(sc_idx, inner[i + k]) for k in range(clip_size)])

    if not clips:
        return 0.5

    mean = _MEAN.cuda()
    std  = _STD.cuda()
    preds = []

    for clip in clips:
        imgs = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        lmks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        _, imgs_align = crop_align_func(lmks, imgs)
        t = torch.as_tensor(imgs_align, dtype=torch.float32).cuda()
        t = t.permute(3, 0, 1, 2).unsqueeze(0).sub(mean).div(std)
        with torch.no_grad():
            out = classifier(t)
        preds.append(float(F.sigmoid(out["final_output"])))

    return float(np.mean(preds))
