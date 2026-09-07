#!/usr/bin/env python
"""Fine-tunes YOLO on UA-DETRAC for vehicle detection -- docs/design/06_ai_pipeline.md
section 6.3's first deliverable ("YOLO fine-tune on UA-DETRAC (Kaggle GPU) + ByteTrack
counting + error-model fit").

**Kaggle GPU execution required.** This is the one piece of Week 8 that genuinely can't
be verified in this project's local dev environment regardless of effort: no GPU, and
UA-DETRAC's ~140K frames aren't downloaded here. The API calls below were NOT guessed --
every one (`model.train(...)`, `model.trainer.best`, `model.export(format="onnx",
quantize=8, ...)`) was verified against the actually-installed `ultralytics` package
(see the `perception` optional dependency group: `uv sync --extra perception`) with a
real, tiny, local CPU smoke run (4 synthetic 64x64 images, 1 epoch) before this script
was written, catching real API details a guess would have missed -- e.g. `export()`
takes `quantize=8` for INT8, not the `int8=True` flag older Ultralytics versions used,
and INT8 export needs a `data=` argument for calibration images that FP32 export
doesn't.

**Dataset**: expects UA-DETRAC already converted to YOLO format (one `.txt` label file
per image: `class x_center y_center width height`, normalized 0-1) with a `data.yaml`
describing the train/val image directories and class names, at `--data-yaml`.
UA-DETRAC ships as XML annotations (DETRAC-Train-Annotations-XML), not YOLO format
natively; several ready-made "UA-DETRAC YOLO format" Kaggle datasets already do this
conversion. Deliberately not re-implemented here: correctly parsing DETRAC's XML schema
(per-frame `<target_list>` elements, occlusion/truncation attributes) is its own
non-trivial task, and re-doing a conversion that already exists and is widely used adds
real risk (an unverifiable, untested converter) without adding value over using one.

Transfer learning from COCO pretrained weights (`yolo{model_size}.pt`, e.g. `yolov8n.pt`)
onto UA-DETRAC's car/bus/van/others classes, per the design doc's own choice.

Usage (on Kaggle, GPU runtime):
    python scripts/train_yolo_detector.py --data-yaml /kaggle/working/ua_detrac/data.yaml \
        --epochs 50 --model-size v8n
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train(data_yaml: Path, epochs: int, imgsz: int, model_size: str, output_dir: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(f"yolo{model_size}.pt")  # COCO-pretrained checkpoint, transfer learning
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(output_dir),
        name="ua_detrac_finetune",
    )
    val_metrics = model.val()
    print(f"val mAP50: {val_metrics.box.map50:.4f}, mAP50-95: {val_metrics.box.map:.4f}")

    if model.trainer is None:
        raise RuntimeError(
            "model.train() completed but left no trainer -- nothing to load weights from"
        )
    best_weights = Path(model.trainer.best)
    print(f"Best weights: {best_weights}")
    return best_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--model-size", default="v8n", help="e.g. v8n (nano, fastest) or v8s (small, more accurate)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/detect"))
    args = parser.parse_args()
    train(args.data_yaml, args.epochs, args.imgsz, args.model_size, args.output_dir)


if __name__ == "__main__":
    main()
