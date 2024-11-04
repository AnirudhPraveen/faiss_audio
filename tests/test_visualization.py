import unittest
import numpy as np
from faiss_audio.visualization import IndexVisualizer
import matplotlib.pyplot as plt

class TestIndexVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = IndexVisualizer()
        self.dimension = 64
        self.n_samples = 1000
        
    def test_compare_indices(self):
        results = self.visualizer.compare_indices(
            dimension=self.dimension,
            n_samples=self.n_samples,
            n_queries=10,
            k=5
        )
        
        self.assertIn('flat', results)
        self.assertIn('build_time', results['flat'])
        self.assertIn('search_time', results['flat'])
        self.assertIn('memory_mb', results['flat'])

    def test_plot_performance_metrics(self):
        results = self.visualizer.compare_indices(
            dimension=self.dimension,
            n_samples=self.n_samples
        )
        
        fig, axes = self.visualizer.plot_performance_metrics(results)
        self.assertEqual(len(axes), 3)
        plt.close(fig)

    def test_plot_recall_comparison(self):
        results = self.visualizer.compare_indices(
            dimension=self.dimension,
            n_samples=self.n_samples
        )
        
        fig, ax = self.visualizer.plot_recall_comparison(
            results,
            ground_truth=results['flat']
        )
        self.assertIsNotNone(ax)
        plt.close(fig)