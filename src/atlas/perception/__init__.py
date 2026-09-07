"""Computer vision pipeline (docs/design/06_ai_pipeline.md section 6.3): virtual-loop
counting and the perception error model are implemented and tested here (no GPU/dataset
needed). YOLO fine-tuning and ONNX export are Kaggle-executed scripts under `scripts/`
(see their own docstrings) -- written correctly against real APIs, not run in this repo's
CI/local environment, the same "correct but unverified locally" treatment already given
to this project's Docker/Redpanda infrastructure.
"""

from atlas.perception.error_model import DetectionTally, ErrorModelStats, fit_error_model
from atlas.perception.virtual_loop import LineCrossingCounter, VirtualLoop, occupancy_fraction

__all__ = [
    "LineCrossingCounter",
    "VirtualLoop",
    "occupancy_fraction",
    "DetectionTally",
    "ErrorModelStats",
    "fit_error_model",
]
