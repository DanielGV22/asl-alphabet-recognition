import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from classifier import MLPClassifier
from utils import load_label_map

def plot_confusion(cm, labels, out_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)

def main():
    labels = load_label_map("data/label_map_filtered.json")

    splits = np.load("data/landmarks_splits.npz")
    X_test = splits["X_test"].astype(np.float32)
    y_test = splits["y_test"].astype(np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPClassifier(in_dim=63, num_classes=len(labels), hidden=256, dropout=0.0).to(device)

    ckpt = torch.load("data/asl_mlp.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(X_test).to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    print(classification_report(y_test, pred, target_names=labels, digits=4))

    cm = confusion_matrix(y_test, pred)
    plot_confusion(cm, labels, out_path="demo_assets/confusion_matrix.png")

if __name__ == "__main__":
    main()
