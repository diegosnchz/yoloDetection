"""
Reproducible equivalence check between YOLOv5 detect.py and app_dashboard pipeline.

Run:
    python -m tests.test_equivalence_detectpy

Optional:
    python -m tests.test_equivalence_detectpy --image-dir dataset_final/images/val --num-images 5 --conf 0.30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def parse_detectpy_top1(label_path: Path, image_w: int, image_h: int, class_names: dict[int, str] | list[str]):
    if not label_path.exists():
        return None

    rows: list[dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        cls_id = int(float(parts[0]))
        x_c, y_c, bw, bh, conf = map(float, parts[1:6])
        x1 = (x_c - bw / 2.0) * image_w
        y1 = (y_c - bh / 2.0) * image_h
        x2 = (x_c + bw / 2.0) * image_w
        y2 = (y_c + bh / 2.0) * image_h
        if isinstance(class_names, dict):
            name = class_names.get(cls_id, "unknown")
        else:
            name = class_names[cls_id] if 0 <= cls_id < len(class_names) else "unknown"
        rows.append(
            {
                "name": name,
                "confidence": conf,
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
            }
        )
    if not rows:
        return None
    rows.sort(key=lambda r: r["confidence"], reverse=True)
    return rows[0]


def run_detectpy(
    python_exe: str,
    repo_root: Path,
    image_path: Path,
    weights_path: Path,
    conf: float,
    project_dir: Path,
    class_names: dict[int, str] | list[str],
):
    labels_dir = project_dir / "detectpy" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_file = labels_dir / f"{image_path.stem}.txt"
    if label_file.exists():
        label_file.unlink()

    cmd = [
        python_exe,
        str(repo_root / "yolov5" / "detect.py"),
        "--weights",
        str(weights_path),
        "--source",
        str(image_path),
        "--conf-thres",
        str(conf),
        "--save-txt",
        "--save-conf",
        "--project",
        str(project_dir),
        "--name",
        "detectpy",
        "--exist-ok",
        "--nosave",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"detect.py failed for {image_path.name}:\n{result.stdout}\n{result.stderr}")

    im = cv2.imread(str(image_path))
    if im is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h, w = im.shape[:2]
    return parse_detectpy_top1(label_file, w, h, class_names)


def run_app_top1(image_path: Path, conf: float, model_bundle):
    from core.pipeline import run_inference_detailed

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _annotated, detections, _raw = run_inference_detailed(
        image_bgr_or_rgb=rgb,
        model_bundle=model_bundle,
        conf_slider=conf,
        iou=0.45,
    )
    if detections.empty:
        return None
    top = detections.sort_values("confidence", ascending=False).iloc[0]
    return {
        "name": str(top["name"]),
        "confidence": float(top["confidence"]),
        "xmin": float(top["xmin"]),
        "ymin": float(top["ymin"]),
        "xmax": float(top["xmax"]),
        "ymax": float(top["ymax"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="dataset_final/images/val")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    image_dir = (repo_root / args.image_dir).resolve()
    weights_path = (repo_root / "yolov5" / "runs_academic" / "actividad1_50e" / "weights" / "best.pt").resolve()
    project_dir = (repo_root / "yolov5" / "runs_test").resolve()

    if not image_dir.exists():
        print(f"[ERROR] image dir not found: {image_dir}")
        return 1
    if not weights_path.exists():
        print(f"[ERROR] weights not found: {weights_path}")
        return 1

    patterns = ("*.jpg", "*.jpeg", "*.png")
    images: list[Path] = []
    for pattern in patterns:
        images.extend(sorted(image_dir.glob(pattern)))
    images = sorted(images)[: args.num_images]
    if not images:
        print(f"[ERROR] no images found in {image_dir}")
        return 1

    # Load the same model bundle used by app pipeline once (pure module, no Streamlit import).
    from core.pipeline import load_model_bundle

    model_bundle = load_model_bundle(weights_path=weights_path)
    class_names = model_bundle[5]

    total = 0
    passed = 0
    failed = 0

    print(f"[INFO] images={len(images)} conf={args.conf}")
    print(f"[INFO] weights={weights_path}")

    for image_path in images:
        total += 1
        try:
            detect_top1 = run_detectpy(
                python_exe=sys.executable,
                repo_root=repo_root,
                image_path=image_path,
                weights_path=weights_path,
                conf=args.conf,
                project_dir=project_dir,
                class_names=class_names,
            )
            app_top1 = run_app_top1(image_path=image_path, conf=args.conf, model_bundle=model_bundle)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {image_path.name} runtime_error={exc}")
            continue

        if detect_top1 is None and app_top1 is None:
            passed += 1
            print(f"[OK]   {image_path.name} both_no_detections")
            continue

        if (detect_top1 is None) != (app_top1 is None):
            failed += 1
            print(
                f"[FAIL] {image_path.name} mismatch_presence "
                f"detectpy={detect_top1 is not None} app={app_top1 is not None}"
            )
            continue

        assert detect_top1 is not None and app_top1 is not None
        same_class = detect_top1["name"] == app_top1["name"]
        conf_diff = abs(float(detect_top1["confidence"]) - float(app_top1["confidence"]))
        iou = iou_xyxy(
            (
                float(detect_top1["xmin"]),
                float(detect_top1["ymin"]),
                float(detect_top1["xmax"]),
                float(detect_top1["ymax"]),
            ),
            (
                float(app_top1["xmin"]),
                float(app_top1["ymin"]),
                float(app_top1["xmax"]),
                float(app_top1["ymax"]),
            ),
        )
        ok = same_class and (conf_diff < 1e-4) and (iou > 0.99)
        if ok:
            passed += 1
            print(
                f"[OK]   {image_path.name} class={app_top1['name']} "
                f"conf_diff={conf_diff:.6f} iou={iou:.6f}"
            )
        else:
            failed += 1
            print(
                f"[FAIL] {image_path.name} class_ok={same_class} "
                f"conf_diff={conf_diff:.6f} iou={iou:.6f} "
                f"detectpy={detect_top1} app={app_top1}"
            )

    print(f"\n[SUMMARY] total={total} passed={passed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
