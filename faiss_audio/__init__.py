from .index import AudioIndex
from .search import AudioSearch
from .utils import get_gpu_resources, index_cpu_to_gpu

__version__ = "0.1.0"

# Define public API
__all__ = [
    "AudioIndex",
    "AudioSearch",
    "get_gpu_resources",
    "index_cpu_to_gpu",
]

# Version info
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0

# Optional GPU support indicator
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

def get_version():
    """Return the current version of faiss-audio."""
    return f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"

def gpu_available():
    """Check if GPU support is available."""
    return GPU_AVAILABLE