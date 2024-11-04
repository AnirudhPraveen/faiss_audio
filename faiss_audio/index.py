import faiss
import numpy as np
from typing import List, Dict, Optional
import json
from pathlib import Path
from .utils import get_gpu_resources, index_cpu_to_gpu

class AudioIndex:
    """A wrapper around FAISS for audio similarity search with optimized PQ."""
    
    def __init__(self, 
                 dimension: int,
                 index_type: str = "ivfpq",
                 pq_config: Optional[Dict] = None,
                 use_gpu: bool = True,
                 gpu_id: Optional[int] = None):
        """
        Initialize the audio index.
        
        Args:
            dimension: Dimensionality of audio embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'ivfpq', 'hnsw')
            pq_config: Configuration for Product Quantization
            use_gpu: Whether to use GPU if available
            gpu_id: Specific GPU to use (None for automatic selection)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # Configure PQ parameters
        self.pq_config = self._get_default_pq_config() if pq_config is None else pq_config
        
        # Initialize GPU resources
        self.gpu_resources = get_gpu_resources() if use_gpu else {}
        
        # Create index
        self.index = self._create_index()
        self.metadata = {}
        self.is_trained = False

    def _get_default_pq_config(self) -> Dict:
        """Get default PQ configuration based on dimension."""
        n_subvectors = min(max(self.dimension // 8, 8), 96)
        n_subvectors = (n_subvectors // 4) * 4  # Make divisible by 4
        
        return {
            'n_subvectors': n_subvectors,
            'bits_per_code': 8,
            'nlist': min(4096, max(64, int(np.sqrt(1000))))
        }

    def _create_index(self) -> faiss.Index:
        """Create the appropriate FAISS index."""
        d = self.dimension

        if self.index_type == "flat":
            cpu_index = faiss.IndexFlatL2(d)
            
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(d)
            cpu_index = faiss.IndexIVFFlat(
                quantizer, d, 
                self.pq_config['nlist']
            )
            
        elif self.index_type == "ivfpq":
            quantizer = faiss.IndexFlatL2(d)
            cpu_index = faiss.IndexIVFPQ(
                quantizer, d,
                self.pq_config['nlist'],
                self.pq_config['n_subvectors'],
                self.pq_config['bits_per_code']
            )
            
        elif self.index_type == "hnsw":
            cpu_index = faiss.IndexHNSWFlat(d, 32)
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # Convert to GPU if requested and available
        if self.use_gpu and self.gpu_resources:
            return index_cpu_to_gpu(cpu_index, self.gpu_id, self.gpu_resources)
        return cpu_index

    def add(self, 
            embeddings: np.ndarray, 
            metadata: Optional[List[Dict]] = None,
            batch_size: int = 10000) -> None:
        """Add embeddings to the index."""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {embeddings.shape[1]}"
            )

        # Process in batches
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]

            # Train index if needed
            if not self.is_trained and self.index_type in ['ivf', 'ivfpq']:
                self.index.train(batch)
                self.is_trained = True

            # Add batch to index
            self.index.add(batch)
            
            # Add metadata
            if metadata:
                batch_metadata = metadata[i:i + batch_size]
                for j, meta in enumerate(batch_metadata):
                    self.metadata[i + j] = meta

    def search(self, 
               query: np.ndarray, 
               k: int = 3, 
               nprobe: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Search for similar embeddings.
        
        Args:
            query: Query embedding(s)
            k: Number of nearest neighbors to return
            nprobe: Number of cells to visit for IVF indices
        """
        # Set search parameters
        if nprobe and self.index_type in ['ivf', 'ivfpq']:
            self.index.nprobe = nprobe

        distances, indices = self.index.search(query, k)
        
        results = {
            'distances': distances,
            'indices': indices,
            'metadata': [[self.metadata.get(idx, {}) for idx in batch] 
                        for batch in indices]
        }
        
        return results

    def save(self, path: str) -> None:
        """Save the index and metadata."""
        path = Path(path)
        # Save the index
        faiss.write_index(faiss.index_gpu_to_cpu(self.index) 
                         if self.use_gpu else self.index, str(path))
        
        # Save metadata and config
        config = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'pq_config': self.pq_config,
            'metadata': self.metadata
        }
        with open(f"{path}.config.json", 'w') as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str, use_gpu: bool = True) -> 'AudioIndex':
        """Load an index from disk."""
        path = Path(path)
        # Load config
        with open(f"{path}.config.json", 'r') as f:
            config = json.load(f)
            
        # Create instance
        instance = cls(
            dimension=config['dimension'],
            index_type=config['index_type'],
            pq_config=config['pq_config'],
            use_gpu=use_gpu
        )
        
        # Load index
        instance.index = faiss.read_index(str(path))
        if use_gpu and instance.gpu_resources:
            instance.index = index_cpu_to_gpu(
                instance.index, 
                instance.gpu_id, 
                instance.gpu_resources
            )
            
        # Load metadata
        instance.metadata = config['metadata']
        instance.is_trained = True
        
        return instance