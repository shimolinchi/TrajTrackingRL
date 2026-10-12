"""Application defaults for the supported CPU/CUDA runtime."""
import os

# Optional third-party device plugins are unused; discovering them scans every
# installed distribution and is particularly expensive on network filesystems.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
