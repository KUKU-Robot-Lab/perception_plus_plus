#!/usr/bin/env python3
"""Build a cup init mask via YOLO bbox + depth banding (lightweight).

Detection: YOLO (detection weights, e.g. yolo11n.pt) gives the cup bounding box
(COCO class 41). Mask: inside the box, keep pixels whose depth lies in a band
around the box's median depth -- this drops the background seen through/around
the cup and needs no segmentation model.
"""
import argparse
import numpy as np

CUP_CLASS = 41  # COCO 'cup'


def yolo_cup_bbox(rgb, weights, conf):
    from ultralytics import YOLO
    model = YOLO(weights)
    best = None
    for res in model(rgb, verbose=False):
        if res.boxes is None:
            continue
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cfd = res.boxes.conf.cpu().numpy()
        box = res.boxes.xyxy.cpu().numpy()
        for i, (c, p) in enumerate(zip(cls, cfd)):
            if c == CUP_CLASS and p >= conf and (best is None or p > best[0]):
                best = (float(p), box[i])
    return best  # (conf, xyxy) or None


def bbox_depth_mask(depth, xyxy, band_m, min_d, max_d):
    h, w = depth.shape
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    mask = np.zeros((h, w), dtype=bool)
    roi = depth[y0:y1, x0:x1]
    valid = roi[(roi > min_d) & (roi < max_d)]
    if valid.size == 0:
        mask[y0:y1, x0:x1] = True  # no depth -> fall back to full box
        return mask, None
    med = float(np.median(valid))
    band = (depth > med - band_m) & (depth < med + band_m)
    box = np.zeros_like(mask)
    box[y0:y1, x0:x1] = True
    mask = box & band
    return mask, med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--band", type=float, default=0.06, help="depth half-band (m)")
    ap.add_argument("--min-d", type=float, default=0.1)
    ap.add_argument("--max-d", type=float, default=3.0)
    ap.add_argument("--preview-mask", default=None)
    args = ap.parse_args()

    d = np.load(args.raw)
    rgb, depth, K = d["rgb"], d["depth"], d["K"]

    det = yolo_cup_bbox(rgb, args.weights, args.conf)
    if det is None:
        print("YOLO found no cup; aborting")
        return 1
    conf, xyxy = det
    mask, med = bbox_depth_mask(depth, xyxy, args.band, args.min_d, args.max_d)
    print(f"cup bbox conf={conf:.3f} xyxy={[round(v,1) for v in xyxy]} "
          f"depth_median={med:.3f}m mask={int(mask.sum())} px")

    np.savez_compressed(args.out, rgb=rgb, depth=depth.astype(np.float32),
                        mask=mask.astype(bool), K=K)
    print(f"wrote {args.out}")

    if args.preview_mask:
        try:
            import cv2
            ov = rgb[:, :, ::-1].copy()
            ov[mask] = (0.5 * ov[mask] + np.array([0, 0, 128])).astype(np.uint8)
            x0, y0, x1, y1 = [int(v) for v in xyxy]
            cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.imwrite(args.preview_mask, ov)
            print(f"mask preview {args.preview_mask}")
        except Exception as e:
            print(f"preview skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
