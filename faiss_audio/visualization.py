import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from time import time
import seaborn as sns
from .index import AudioIndex
from .search import AudioSearch

class IndexVisualizer:
    """Visualization tools for analyzing FAISS index performance."""
    
    def __init__(self):
        self.index_types = ['flat', 'ivf', 'ivfpq', 'hnsw']
        self.index_properties = {
            'flat': {
                'name': 'Flat L2',
                'color': '#FF9999',
                'description': 'Exact search, best for small datasets'
            },
            'ivf': {
                'name': 'IVF',
                'color': '#66B2FF',
                'description': 'Good balance for medium datasets'
            },
            'ivfpq': {
                'name': 'IVF-PQ',
                'color': '#99FF99',
                'description': 'Best for large datasets'
            },
            'hnsw': {
                'name': 'HNSW',
                'color': '#FFCC99',
                'description': 'High accuracy, fast search'
            }
        }

    def compare_indices(self,
                       dimension: int,
                       n_samples: int,
                       n_queries: int = 100,
                       k: int = 5) -> Dict:
        """Compare different index types using sample data."""
        embeddings = np.random.random((n_samples, dimension)).astype('float32')
        queries = np.random.random((n_queries, dimension)).astype('float32')
        
        results = {}
        for idx_type in self.index_types:
            print(f"Benchmarking {idx_type}...")
            
            # Time index creation and training
            start_time = time()
            index = AudioIndex(dimension=dimension, index_type=idx_type)
            index.add(embeddings)
            build_time = time() - start_time
            
            # Time search
            start_time = time()
            search_results = index.search(queries, k=k)
            search_time = time() - start_time
            
            # Estimate memory usage
            memory_usage = index.index.ntotal * dimension * 4
            
            results[idx_type] = {
                'build_time': build_time,
                'search_time': search_time,
                'memory_mb': memory_usage / (1024 * 1024),
                'results': search_results
            }
        
        return results

    def plot_performance_metrics(self,
                               benchmark_results: Dict,
                               figsize: Tuple[int, int] = (15, 5)):
        """Plot performance metrics for different index types."""
        metrics = ['build_time', 'search_time', 'memory_mb']
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        
        metric_names = {
            'build_time': 'Build Time (s)',
            'search_time': 'Search Time (s)',
            'memory_mb': 'Memory Usage (MB)'
        }
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [benchmark_results[idx][metric] for idx in self.index_types]
            colors = [self.index_properties[idx]['color'] for idx in self.index_types]
            names = [self.index_properties[idx]['name'] for idx in self.index_types]
            
            bars = ax.bar(names, values, color=colors)
            ax.set_title(metric_names.get(metric, metric))
            ax.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig, axes

    def plot_recall_comparison(self,
                             benchmark_results: Dict,
                             ground_truth: Dict,
                             figsize: Tuple[int, int] = (10, 6)):
        """Plot recall accuracy comparison."""
        plt.figure(figsize=figsize)
        
        recalls = {}
        for idx_type in self.index_types:
            if idx_type == 'flat':
                recalls[idx_type] = 1.0
                continue
                
            results = benchmark_results[idx_type]['results']['indices']
            gt = ground_truth['results']['indices']
            recall = np.mean([
                len(set(r) & set(g)) / len(g)
                for r, g in zip(results, gt)
            ])
            recalls[idx_type] = recall
        
        colors = [self.index_properties[idx]['color'] for idx in self.index_types]
        names = [self.index_properties[idx]['name'] for idx in self.index_types]
        
        bars = plt.bar(names, recalls.values(), color=colors)
        plt.title('Recall Accuracy Comparison')
        plt.ylabel('Recall@k')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        return plt.gcf(), plt.gca()