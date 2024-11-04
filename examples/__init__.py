import os
import sys

# Add parent directory to path to allow direct running of examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Utility functions for examples
def create_sample_audio(filename, frequency, duration=3, sample_rate=16000):
    """Create a sample audio file with given frequency."""
    import numpy as np
    import soundfile as sf
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    sf.write(filename, audio_data, sample_rate)
    return filename

def create_example_dataset(output_dir, n_files=5):
    """Create example dataset for demonstrations."""
    import numpy as np
    from pathlib import Path
    
    # Create directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create files with different frequencies
    files = []
    for i in range(n_files):
        frequency = 440 * (i + 1)  # A4, A5, etc.
        filename = output_dir / f"audio_{i}.wav"
        files.append(create_sample_audio(filename, frequency))
    
    return files