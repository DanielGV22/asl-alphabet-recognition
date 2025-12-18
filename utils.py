import json
import numpy as np

def normalize_landmarks(pts_21x3: np.ndarray) -> np.ndarray:
    pts = pts_21x3.copy()

    # Translation normalize (wrist as origin)
    wrist = pts[0].copy()
    pts -= wrist

    # Scale normalize (max xy distance)
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale

    return pts.reshape(-1).astype(np.float32)  # (63,)

def bbox_norm_to_pixels(bbox_norm, w, h, pad=0.02):
    x0, y0, x1, y1 = bbox_norm
    x0 = max(0.0, x0 - pad); y0 = max(0.0, y0 - pad)
    x1 = min(1.0, x1 + pad); y1 = min(1.0, y1 + pad)
    return int(x0*w), int(y0*h), int(x1*w), int(y1*h)

def save_label_map(path, labels):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, indent=2)

def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["labels"]
