import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from hand_detector import HandDetector
from classifier import MLPClassifier, predict
from utils import normalize_landmarks, bbox_norm_to_pixels, load_label_map
from metrics import PerfMeter, gpu_mem_mb
from overlay import draw_bbox, draw_hud

def load_model(model_path, device):
    # IMPORTANT: use filtered label map produced by training
    labels = load_label_map("data/label_map_filtered.json")
    model = MLPClassifier(in_dim=63, num_classes=len(labels), hidden=256, dropout=0.0).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, labels

def process_stream(cap, model, labels, device, save_path=None):
    det = HandDetector(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    meter = PerfMeter(fps_window=30)

    pred_hist = deque(maxlen=15)
    prob_hist = deque(maxlen=15)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        meter.tick()

        t0 = time.perf_counter()
        res = det.detect(frame)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        if res is not None:
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = bbox_norm_to_pixels(res["bbox_norm"], w, h)
            hand_conf = res["hand_conf"]

            feats = normalize_landmarks(res["landmarks"])
            cls_idx, cls_prob = predict(model, feats, device)

            pred_hist.append(cls_idx)
            prob_hist.append(cls_prob)

            vals, counts = np.unique(np.array(pred_hist), return_counts=True)
            sm_idx = int(vals[int(np.argmax(counts))])
            sm_prob = float(np.mean(prob_hist))

            letter = labels[sm_idx]
            draw_bbox(frame, x0, y0, x1, y1, f"HAND {hand_conf:.2f} | {letter} {sm_prob:.2f}")

        cpu = meter.cpu_percent()
        ram = meter.ram_mb()
        fps = meter.fps()
        gmem = gpu_mem_mb()

        hud = [
            f"FPS: {fps:.1f}",
            f"Infer(ms): {infer_ms:.1f}",
            f"CPU%: {cpu:.1f}",
            f"RAM(MB): {ram:.1f}",
        ]
        if gmem is not None:
            hud.append(f"GPU Mem(MB): {gmem:.1f}")

        draw_hud(frame, hud)

        cv2.imshow("ASL Alphabet (Detection + Classification)", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    if writer:
        writer.release()

def process_image(path, model, labels, device, save_path=None):
    det = HandDetector(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    res = det.detect(img)
    if res is not None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = bbox_norm_to_pixels(res["bbox_norm"], w, h)
        feats = normalize_landmarks(res["landmarks"])
        idx, prob = predict(model, feats, device)
        draw_bbox(img, x0, y0, x1, y1, f"HAND {res['hand_conf']:.2f} | {labels[idx]} {prob:.2f}")

    if save_path:
        cv2.imwrite(save_path, img)
        print("Saved:", save_path)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["webcam", "video", "image"], required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--model", type=str, default="data/asl_mlp.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, labels = load_model(args.model, device)

    if args.mode == "webcam":
        cap = cv2.VideoCapture(args.camera)
        process_stream(cap, model, labels, device, save_path=args.save)
        cap.release()

    elif args.mode == "video":
        if not args.input:
            raise ValueError("--input is required for video mode")
        cap = cv2.VideoCapture(args.input)
        process_stream(cap, model, labels, device, save_path=args.save)
        cap.release()

    elif args.mode == "image":
        if not args.input:
            raise ValueError("--input is required for image mode")
        process_image(args.input, model, labels, device, save_path=args.save)

if __name__ == "__main__":
    main()
