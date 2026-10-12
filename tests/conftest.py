"""Bound CPU tests to avoid oversubscribing shared training nodes."""
import os
from pathlib import Path

(Path(__file__).resolve().parents[1] / "tmp").mkdir(exist_ok=True)
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch

torch.set_num_threads(2)
