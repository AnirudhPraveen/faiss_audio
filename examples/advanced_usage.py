import torch
import numpy as np
from pathlib import Path
from faiss_audio import AudioSearch
from examples import create_example_dataset
import time

def compare_index_types():
    """Compare different index types."""
    dataset_dir = Path("example_dataset")
    audio_files = create_example_dataset(dataset_dir)
    query_file = "query.wav"
    
    # Test different index types
    index_types = ["flat", "ivf", "ivfpq", "hnsw"]
    results = {}
    
    for index_type in index_types:
        print(f"\nTesting {index_type} index...")
        
        # Configure search based on index type
        config = {}
        if index_type == "ivfpq":
            config = {
                'n_subvectors': 16,
                'bits_per_code': 8,
                'nlist': 100
            }
        elif index_type == "ivf":
            config = {'nlist': 100}
            
        # Initialize search
        search = AudioSearch(
            index_type=index_type,
            pq_config=config,
            use_gpu=True
        )
        
        # Time indexing
        import time
        start_time = time.time()
        search.index_audio_files(str(dataset_dir))
        index_time = time.time() - start_time
        
        # Time search
        start_time = time.time()
        search_results = search.search(query_file, k=3)
        search_time = time.time() - start_time
        
        results[index_type] = {
            'index_time': index_time,
            'search_time': search_time,
            'results': search_results
        }
        
    # Print comparison
    print("\nPerformance Comparison:")
    print("\nIndex Type | Index Time | Search Time | Top Match Distance")
    print("-" * 60)
    for idx_type, data in results.items():
        print(f"{idx_type:9} | {data['index_time']:10.3f} | {data['search_time']:10.3f} | {data['results'][0]['distance']:.3f}")

def gpu_vs_cpu_comparison():
    """Compare GPU vs CPU performance."""
    if not torch.cuda.is_available():
        print("GPU not available, skipping comparison")
        return
        
    dataset_dir = Path("example_dataset")
    audio_files = create_example_dataset(dataset_dir)
    query_file = "query.wav"
    
    # Test both GPU and CPU
    for use_gpu in [True, False]:
        device = "GPU" if use_gpu else "CPU"
        print(f"\nTesting on {device}...")
        
        search = AudioSearch(use_gpu=use_gpu)
        
        # Time indexing
        start_time = time.time()
        search.index_audio_files(str(dataset_dir))
        index_time = time.time() - start_time
        
        # Time search
        start_time = time.time()
        results = search.search(query_file, k=3)
        search_time = time.time() - start_time
        
        print(f"{device} Index Time: {index_time:.3f}s")
        print(f"{device} Search Time: {search_time:.3f}s")
        print(f"{device} Top Match Distance: {results[0]['distance']:.3f}")

def batch_size_experiment():
    """Experiment with different batch sizes."""
    dataset_dir = Path("example_dataset")
    audio_files = create_example_dataset(dataset_dir, n_files=100)  # More files
    
    batch_sizes = [1, 10, 32, 64, 128]
    
    print("\nBatch Size Comparison:")
    for batch_size in batch_sizes:
        search = AudioSearch()
        
        start_time = time.time()
        search.index_audio_files(
            str(dataset_dir),
            batch_size=batch_size,
            show_progress=False
        )
        index_time = time.time() - start_time
        
        print(f"Batch Size {batch_size:3d}: {index_time:.3f}s")

def main():
    """Run advanced usage examples."""
    print("1. Comparing Index Types")
    compare_index_types()
    
    print("\n2. GPU vs CPU Comparison")
    gpu_vs_cpu_comparison()
    
    print("\n3. Batch Size Experiment")
    batch_size_experiment()

if __name__ == "__main__":
    main()