import unittest
import numpy as np
import tempfile
from pathlib import Path
from faiss_audio import AudioIndex

class TestAudioIndex(unittest.TestCase):
    def setUp(self):
        self.dimension = 128
        self.n_vectors = 1000
        self.index = AudioIndex(dimension=self.dimension)
        self.vectors = np.random.random((self.n_vectors, self.dimension)).astype('float32')
        
    def test_add_and_search(self):
        metadata = [{"id": i} for i in range(self.n_vectors)]
        self.index.add(self.vectors, metadata)
        
        query = np.random.random((1, self.dimension)).astype('float32')
        results = self.index.search(query, k=5)
        
        self.assertEqual(results['distances'].shape, (1, 5))
        self.assertEqual(results['indices'].shape, (1, 5))
        self.assertEqual(len(results['metadata'][0]), 5)

    def test_save_load(self):
        self.index.add(self.vectors)
        
        with tempfile.NamedTemporaryFile() as tmp:
            path = Path(tmp.name)
            self.index.save(path)
            loaded_index = AudioIndex.load(path)
            
            query = np.random.random((1, self.dimension)).astype('float32')
            original_results = self.index.search(query)
            loaded_results = loaded_index.search(query)
            
            np.testing.assert_array_almost_equal(
                original_results['distances'],
                loaded_results['distances']
            )