import unittest
import numpy as np
import tempfile
from pathlib import Path
import soundfile as sf
from faiss_audio import AudioSearch

class TestAudioSearch(unittest.TestCase):
    def setUp(self):
        self.search = AudioSearch(use_gpu=False)
        
        # Create test audio files
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []
        
        for i in range(3):
            file_path = Path(self.temp_dir) / f"test_{i}.wav"
            sample_rate = 16000
            t = np.linspace(0, 1, sample_rate)
            audio_data = np.sin(2 * np.pi * (440 * (i + 1)) * t)
            sf.write(file_path, audio_data, sample_rate)
            self.test_files.append(file_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_index_audio_files(self):
        self.search.index_audio_files(self.test_files)
        metadata = self.search.index.metadata
        self.assertEqual(len(metadata), len(self.test_files))

    def test_search(self):
        self.search.index_audio_files(self.test_files)
        results = self.search.search(self.test_files[0], k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIn('distance', results[0])
        self.assertIn('filename', results[0])
        self.assertIn('path', results[0])

    def test_save_load(self):
        self.search.index_audio_files(self.test_files)
        
        with tempfile.NamedTemporaryFile() as tmp:
            path = Path(tmp.name)
            self.search.save(path)
            
            loaded_search = AudioSearch.load(path, use_gpu=False)
            original_results = self.search.search(self.test_files[0])
            loaded_results = loaded_search.search(self.test_files[0])
            
            self.assertEqual(len(original_results), len(loaded_results))