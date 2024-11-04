import unittest
import numpy as np
import faiss
import torch
from faiss_audio.utils import get_gpu_resources, index_cpu_to_gpu

class TestUtils(unittest.TestCase):
    def setUp(self):
        """Initialize test environment."""
        # Create sample data for index testing
        self.dimension = 64
        self.n_vectors = 1000
        self.vectors = np.random.random((self.n_vectors, self.dimension)).astype('float32')

    def test_get_gpu_resources(self):
        """Test GPU resource acquisition."""
        resources = get_gpu_resources()
        
        if torch.cuda.is_available():
            # If GPU is available
            self.assertGreater(len(resources), 0)
            for res in resources.values():
                # Verify resource type
                self.assertIsInstance(res, faiss.GpuResources)
                # Verify memory allocation
                self.assertTrue(hasattr(res, 'getMemoryAvailable'))
        else:
            # If no GPU, should return empty dict
            self.assertEqual(len(resources), {})

    def test_index_cpu_to_gpu_with_gpu(self):
        """Test CPU to GPU index conversion when GPU is available."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")

        # Create CPU index
        cpu_index = faiss.IndexFlatL2(self.dimension)
        cpu_index.add(self.vectors)

        # Get resources
        resources = get_gpu_resources()
        
        # Convert to GPU
        gpu_index = index_cpu_to_gpu(cpu_index, gpu_id=None, resources=resources)
        
        # Verify conversion
        self.assertIsInstance(gpu_index, faiss.GpuIndex)
        
        # Test functionality
        query = np.random.random((1, self.dimension)).astype('float32')
        cpu_distances, cpu_indices = cpu_index.search(query, k=5)
        gpu_distances, gpu_indices = gpu_index.search(query, k=5)
        
        # Results should be similar (not exact due to float16)
        np.testing.assert_array_almost_equal(cpu_distances, gpu_distances, decimal=3)

    def test_index_cpu_to_gpu_no_gpu(self):
        """Test CPU to GPU conversion when no GPU is available."""
        # Create CPU index
        cpu_index = faiss.IndexFlatL2(self.dimension)
        cpu_index.add(self.vectors)
        
        # Try conversion with no resources
        result_index = index_cpu_to_gpu(cpu_index, resources=None)
        
        # Should return original CPU index
        self.assertEqual(type(result_index), type(cpu_index))

    def test_index_cpu_to_gpu_invalid_gpu_id(self):
        """Test handling of invalid GPU ID."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")

        # Create CPU index
        cpu_index = faiss.IndexFlatL2(self.dimension)
        resources = get_gpu_resources()
        
        # Try with invalid GPU ID
        invalid_gpu_id = len(resources) + 1
        gpu_index = index_cpu_to_gpu(cpu_index, gpu_id=invalid_gpu_id, resources=resources)
        
        # Should still work by falling back to available GPU
        self.assertIsNotNone(gpu_index)

    def test_multiple_gpu_handling(self):
        """Test handling of multiple GPUs if available."""
        resources = get_gpu_resources()
        if len(resources) <= 1:
            self.skipTest("Multiple GPUs not available")

        # Create CPU index
        cpu_index = faiss.IndexFlatL2(self.dimension)
        cpu_index.add(self.vectors)

        # Test on each available GPU
        for gpu_id in range(len(resources)):
            gpu_index = index_cpu_to_gpu(cpu_index, gpu_id=gpu_id, resources=resources)
            
            # Verify conversion and functionality
            self.assertIsInstance(gpu_index, faiss.GpuIndex)
            query = np.random.random((1, self.dimension)).astype('float32')
            distances, indices = gpu_index.search(query, k=5)
            self.assertEqual(distances.shape, (1, 5))

    def test_memory_cleanup(self):
        """Test proper GPU memory cleanup."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")

        resources = get_gpu_resources()
        initial_memory = torch.cuda.memory_allocated()

        # Create and convert index
        cpu_index = faiss.IndexFlatL2(self.dimension)
        cpu_index.add(self.vectors)
        gpu_index = index_cpu_to_gpu(cpu_index, resources=resources)

        # Use the index
        query = np.random.random((1, self.dimension)).astype('float32')
        gpu_index.search(query, k=5)

        # Clean up
        del gpu_index
        torch.cuda.empty_cache()

        # Check memory
        final_memory = torch.cuda.memory_allocated()
        self.assertLessEqual(final_memory, initial_memory)

if __name__ == '__main__':
    unittest.main()