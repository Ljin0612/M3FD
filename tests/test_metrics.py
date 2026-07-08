import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo.tools.metrics import compute_ap


def test_compute_ap_smoke_returns_float():
    recall = np.array([0.0, 0.5, 1.0], dtype=float)
    precision = np.array([1.0, 0.75, 0.5], dtype=float)

    ap = compute_ap(recall, precision)

    assert isinstance(ap, float)
