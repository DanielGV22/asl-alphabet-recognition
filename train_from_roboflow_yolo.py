import os
import glob
import json
import yaml
import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from hand_detector import HandDetector
from utils import normalize_landmarks, save_label_map
from classifier import MLPClassifier

# ---------------------------
# YOLO helpers
# ---------------------------

def read_data_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", None)
    if names is None:
        raise ValueError("data.yaml missing 'names' list.")
    return names

def parse_yolo_label(label_path):
    """
    YOLO format lines:
    class_id x_center y_center width height
    all normalized (0..1)
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            boxes.append((cls_id, x, y, w, h))
    return boxes

def yolo_box_to_pixels(x, y, w, h, img_w, img_h, pad=0.30):
    """
    pad expands the crop. Larger pad improves MediaPipe success rate.
    """
    w2 = w * (1.0 + pad)
    h2 = h * (1.0 + pad)

    x0 = (x - w2 / 2.0) * img_w
    y0 = (y - h2 / 2.0) * img_h
    x1 = (x + w2 / 2.0) * img_w
    y1 = (y + h2 / 2.0) * img_h

    x0 = int(max(0, min(img_w - 1, x0)))
    y0 = int(max(0, min(img_h - 1, y0)))
    x1 = int(max(0, min(img_w - 1, x1)))
    y1 = int(max(0, min(img_h - 1, y1)))

    # Ensure non-empty crop
    if x1 <= x0:
        x1 = min(img_w - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(img_h - 1, y0 + 1)

    return x0, y0, x1, y1

def pick_best_box(boxes):
    # Choose the largest area box (often the hand)
    return max(boxes, key=lambda b: b[3] * b[4])

# ---------------------------
# Landmark extraction
# ---------------------------

def build_landmarks_from_roboflow(
    roboflow_root,
    out_npz="data/landmarks_all.npz",
    label_map_path="data/label_map.json"
):
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    yaml_path = os.path.join(roboflow_root, "data.yaml")
    names = read_data_yaml(yaml_path)
    print("Classes:", names)

    # Save original label map (dataset order)
    save_label_map(label_map_path, names)

    det = HandDetector(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    X, y = [], []
    skipped = 0
    total = 0

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(roboflow_root, split, "images")
        lab_dir = os.path.join(roboflow_root, split, "labels")

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        if not img_paths:
            print(f"Warning: no images found in {img_dir}")

        for img_path in tqdm(img_paths, desc=f"Extract {split}"):
            total += 1

            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(lab_dir, base + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            boxes = parse_yolo_label(label_path)
            if not boxes:
                skipped += 1
                continue

            cls_id, xc, yc, bw, bh = pick_best_box(boxes)

            h, w = img.shape[:2]
            # IMPORTANT: increased pad from 0.15 -> 0.30
            x0, y0, x1, y1 = yolo_box_to_pixels(xc, yc, bw, bh, w, h, pad=0.30)
            crop = img[y0:y1, x0:x1]

            res = det.detect(crop)
            if res is None:
                skipped += 1
                continue

            feats = normalize_landmarks(res["landmarks"])  # (63,)
            X.append(feats)
            y.append(cls_id)

    if not X:
        raise RuntimeError(
            "No landmarks extracted. Common causes:\n"
            "- Wrong ROBOFLOW_ROOT path\n"
            "- Folder layout mismatch\n"
            "- MediaPipe failing due to bad crops\n"
        )

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    np.savez(out_npz, X=X, y=y)
    print(f"Saved: {out_npz}")
    print(f"Total images scanned: {total}")
    print(f"Extracted samples: {len(y)}")
    print(f"Skipped: {skipped}")

    return names

# ---------------------------
# Training
# ---------------------------

class NpzDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_mlp(
    npz_path="data/landmarks_all.npz",
    out_model="data/asl_mlp.pt",
    epochs=25,
    batch=256,
    lr=1e-3,
    min_samples_per_class=2
):
   data = np.load(npz_path)
X, y = data["X"], data["y"]

# --- FIX: handle classes with too few samples for stratified split ---
import json
with open("data/label_map.json", "r", encoding="utf-8") as f:
    full_labels = json.load(f)["labels"]

counts = np.bincount(y, minlength=len(full_labels))
min_required = 2  # must be >= 2 for stratify to work

valid_classes = np.where(counts >= min_required)[0]
dropped_classes = np.where((counts > 0) & (counts < min_required))[0]

print("\nClass counts (extracted):")
for i, c in enumerate(counts):
    if c > 0:
        print(f"  {i:02d} {full_labels[i]}: {int(c)}")

if len(dropped_classes) > 0:
    print("\nDropping classes with too few samples:")
    for i in dropped_classes:
        print(f"  {i:02d} {full_labels[i]}: {int(counts[i])} (dropped)")

# filter samples to valid classes
mask = np.isin(y, valid_classes)
X = X[mask]
y = y[mask]

# re-index class IDs to 0..K-1
old_to_new = {int(old): int(new) for new, old in enumerate(valid_classes)}
y = np.array([old_to_new[int(v)] for v in y], dtype=np.int64)

# save filtered label map (used by main/eval)
filtered_labels = [full_labels[int(i)] for i in valid_classes]
with open("data/label_map_filtered.json", "w", encoding="utf-8") as f:
    json.dump({"labels": filtered_labels}, f, indent=2)

num_classes = len(filtered_labels)
print(f"\nTraining on {num_classes} classes after filtering.")
print("Saved: data/label_map_filtered.json")

# stratified split now guaranteed valid
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = MLPClassifier(in_dim=63, num_classes=num_classes, hidden=256, dropout=0.25).to(device)


train_loader = DataLoader(NpzDataset(X_train, y_train), batch_size=batch, shuffle=True, drop_last=False)
val_loader = DataLoader(NpzDataset(X_val, y_val), batch_size=batch, shuffle=False, drop_last=False)

opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

best_val = 0.0
for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        # Validation
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(yb.numpy())

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        val_acc = accuracy_score(all_true, all_pred)

        print(f"Epoch {ep:02d} | loss={np.mean(losses):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(os.path.dirname(out_model), exist_ok=True)
            torch.save({"state_dict": model.state_dict()}, out_model)
            print("  Saved best model ->", out_model)

np.savez(
        "data/landmarks_splits.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
print("Saved splits -> data/landmarks_splits.npz")

# ---------------------------
# Entry point
# ---------------------------

if __name__ == "__main__":
    ROBOFLOW_ROOT = "data/roboflow_asl"

    build_landmarks_from_roboflow(
        ROBOFLOW_ROOT,
        out_npz="data/landmarks_all.npz",
        label_map_path="data/label_map.json"
    )

    train_mlp(
        npz_path="data/landmarks_all.npz",
        out_model="data/asl_mlp.pt",
        epochs=25,
        batch=256,
        lr=1e-3,
        min_samples_per_class=2
    )
