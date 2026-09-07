#!/usr/bin/env python
"""ONNX export + INT8 quantization + latency/mAP bench (CPU: full vs INT8) -- the "Edge
AI claim, made concrete without owning a Jetson" (docs/design/06_ai_pipeline.md section
6.3, deliverable 4).

Unlike scripts/train_yolo_detector.py, this script's export/val/inference steps need no
GPU -- verified for real on this dev machine with a tiny synthetic checkpoint (4 images,
1 CPU-trained epoch): FP32 and INT8 ONNX export, `.val()` on each exported model, and a
manual latency benchmark all ran correctly here. What's NOT run here is meaningful
mAP/latency numbers, which need a real fine-tuned checkpoint from
scripts/train_yolo_detector.py (Kaggle-only) and a real validation set -- this script is
ready to produce that table the moment such a checkpoint exists.

Two real API details this script depends on, found (not guessed) via that smoke run:
1. `model.export(format="onnx", quantize=8, ...)` needs a `data=<data.yaml>` argument
   for INT8 calibration images -- FP32 export doesn't need this. Ultralytics itself
   warns if fewer than ~300 calibration images are available; pass a real validation
   split, not a handful of images, for a trustworthy INT8 model.
2. Exporting INT8 from the SAME checkpoint a moment after FP32 can overwrite/remove the
   FP32 `.onnx` file on disk (observed directly, not documented behavior to rely on) --
   this script copies the FP32 export to a distinct filename immediately, before ever
   requesting the INT8 export.

**Honest caveat, observed on the local smoke run, not assumed away**: INT8 was smaller
(as expected) but NOT faster than FP32 on this CPU (an 11th-gen Intel laptop chip) --
ONNX Runtime's default CPU execution provider doesn't reliably accelerate INT8 ops
without hardware VNNI support and/or further graph optimization the default export
doesn't apply. The "edge AI claim" this script exists to make concrete is therefore
size, not blindly assumed speed -- report whatever the real numbers say for the actual
target hardware, don't presume INT8 always wins.

Usage:
    python scripts/export_onnx_int8.py --weights runs/detect/ua_detrac_finetune/weights/best.pt \
        --data-yaml /kaggle/working/ua_detrac/data.yaml
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

NUM_WARMUP = 5
NUM_TIMED = 50


def export_fp32(weights: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", simplify=True))
    safe_path = exported.with_name(exported.stem + "_fp32.onnx")
    shutil.copy(exported, safe_path)
    return safe_path


def export_int8(weights: Path, data_yaml: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return Path(model.export(format="onnx", quantize=8, simplify=True, data=str(data_yaml)))


def measure_latency_ms(onnx_path: Path, imgsz: int) -> float:
    from ultralytics import YOLO

    model = YOLO(str(onnx_path))
    dummy_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(NUM_WARMUP):
        model.predict(dummy_image, verbose=False)

    start = time.perf_counter()
    for _ in range(NUM_TIMED):
        model.predict(dummy_image, verbose=False)
    elapsed_s = time.perf_counter() - start
    return (elapsed_s / NUM_TIMED) * 1000


def measure_map50(onnx_path: Path, data_yaml: Path) -> float:
    from ultralytics import YOLO

    model = YOLO(str(onnx_path))
    metrics: Any = model.val(data=str(data_yaml), verbose=False)
    return float(metrics.box.map50)


def benchmark(weights: Path, data_yaml: Path, imgsz: int) -> dict[str, dict[str, float]]:
    fp32_path = export_fp32(weights)
    int8_path = export_int8(weights, data_yaml)

    results = {}
    for label, path in (("fp32", fp32_path), ("int8", int8_path)):
        results[label] = {
            "map50": measure_map50(path, data_yaml),
            "latency_ms": measure_latency_ms(path, imgsz),
            "size_mb": path.stat().st_size / 1e6,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    results = benchmark(args.weights, args.data_yaml, args.imgsz)

    print(f"\n{'format':<10}{'mAP50':>10}{'latency_ms':>14}{'size_mb':>12}")
    for label, metrics in results.items():
        print(
            f"{label:<10}{metrics['map50']:>10.4f}"
            f"{metrics['latency_ms']:>14.2f}{metrics['size_mb']:>12.2f}"
        )

    fp32, int8 = results["fp32"], results["int8"]
    speedup = fp32["latency_ms"] / int8["latency_ms"] if int8["latency_ms"] > 0 else float("nan")
    size_reduction = 1 - int8["size_mb"] / fp32["size_mb"] if fp32["size_mb"] > 0 else float("nan")
    map_drop = fp32["map50"] - int8["map50"]
    print(
        f"\nINT8 vs FP32: {speedup:.2f}x faster, {size_reduction:.0%} smaller, mAP50 drop {map_drop:+.4f}"
    )


if __name__ == "__main__":
    main()
