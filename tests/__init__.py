import os
import sys

# Add parent directory to path to allow importing faiss_audio during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test utilities
def get_test_data_path():
    """Get path to test data directory."""
    return os.path.join(os.path.dirname(__file__), 'data')

def create_test_data_dir():
    """Create test data directory if it doesn't exist."""
    data_dir = get_test_data_path()
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

# Test configuration
TEST_DIMENSION = 128
TEST_NUM_VECTORS = 1000

# Setup common test variables
SAMPLE_RATE = 16000
TEST_AUDIO_DURATION = 3  # seconds

# Define test device
import torch
TEST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"