import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf
from faiss_audio.features import AudioFeatureExtractor

class TestAudioFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.feature_extractor = AudioFeatureExtractor()
        
        # Create test audio
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = Path(self.temp_dir) / "test_audio.wav"
        
        # Generate test audio (1-second sine wave)
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate)
        audio_data = np.sin(2 * np.pi * 440 * t)
        sf.write(self.test_audio_path, audio_data, sample_rate)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_process_audio(self):
        waveform = self.feature_extractor.process_audio(self.test_audio_path)
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertEqual(waveform.shape[0], 1)

    def test_extract_features(self):
        features = self.feature_extractor.extract_features(self.test_audio_path)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[1], self.feature_extractor.embedding_dim)