import faiss
import numpy as np
from typing import List, Dict, Optional
import json
from pathlib import Path
from .utils import get_gpu_resources, index_cpu_to_gpu

class AudioIndex:
    """A wrapper around FAISS for audio similarity search with optimized Product Quantization (PQ)."""
    
    def __init__(self, 
                 dimension: int,
                 index_type: str = "ivfpq",
                 pq_config: Optional[Dict] = None,
                 use_gpu: bool = True,
                 gpu_id: Optional[int] = None):
        """
        Initialize the audio index.
        
        Args:
            dimension (int): Dimensionality of audio embeddings 
            index_type (str): Type of FAISS index ('flat', 'ivf', 'ivfpq', 'hnsw'). Default is ivfpq
            pq_config (Optional[Dict]): Configuration for Product Quantization
            use_gpu (bool): Whether to use GPU if available
            gpu_id (Optional[int]): Specific GPU to use (None for automatic selection)
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
        """Get default Product Quantization (PQ) configuration parameters based on vector dimension.

        This method calculates the optimal number of subvectors based on the dimension,
        and returns a configuration dictionary for PQ indexing.

        Returns:
            Dict: Configuration dictionary containing:
                - n_subvectors (int): Number of subvectors, calculated as dimension/8,
                bounded between 8 and 96, and made divisible by 4
                - bits_per_code (int): Number of bits per subvector code, fixed at 8
                - nlist (int): Number of clusters/cells, calculated as min(4096, max(64, sqrt(1000)))

        Example:
            >>> # For a dimension of 128
            >>> config = get_default_pq_config()
            >>> # Returns something like:
            >>> # {'n_subvectors': 16, 'bits_per_code': 8, 'nlist': 64}
        """
        n_subvectors = min(max(self.dimension // 8, 8), 96)
        n_subvectors = (n_subvectors // 4) * 4  # Make divisible by 4
        
        return {
            'n_subvectors': n_subvectors,
            'bits_per_code': 8,
            'nlist': min(4096, max(64, int(np.sqrt(1000))))
        }

    def _create_index(self) -> faiss.Index:
        """Create and initialize a FAISS index based on specified configuration.

        This method creates one of several types of FAISS indices based on self.index_type:
        - "flat": Basic exhaustive search index using L2 distance
        - "ivf": Inverted File index with flat L2 quantizer
        - "ivfpq": IVF index with Product Quantization for compressed storage
        - "hnsw": Hierarchical Navigable Small World graph-based index

        Returns:
            faiss.Index: The initialized FAISS index object. If use_gpu=True and GPU resources
            are available, returns a GPU-enabled version of the index.

        Raises:
            ValueError: If an unsupported index_type is specified

        Example:
            >>> # Create a flat index for 128-dimensional vectors
            >>> index = create_index()  # with self.index_type="flat"
            >>> # Returns a faiss.IndexFlatL2 object (or GPU version if enabled)
        
        Notes:
            - IVF indices use a flat quantizer internally
            - HNSW index uses 32 neighbors by default
            - PQ configuration (nlist, n_subvectors, bits_per_code) is read from self.pq_config
            - GPU conversion happens automatically if self.use_gpu=True and resources available
        """
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
        """Add embeddings and optional metadata to the FAISS index in batches.

        This method adds embedding vectors to the index and associates metadata with them.
        For IVF-based indices (ivf, ivfpq), it also handles training if not already trained.
        Processing is done in batches to manage memory usage.

        Args:
            embeddings (np.ndarray): Array of embedding vectors to add, shape (n_vectors, dimension)
            metadata (Optional[List[Dict]], optional): List of metadata dictionaries for each vector.
                Must be same length as embeddings if provided. Defaults to None.
            batch_size (int, optional): Number of vectors to process in each batch. 
                Defaults to 10000.

        Raises:
            ValueError: If embeddings dimension doesn't match index dimension

        Example:
            >>> # Add 1000 vectors of dimension 128
            >>> vectors = np.random.random((1000, 128))
            >>> metadata = [{'id': i} for i in range(1000)]
            >>> add(vectors, metadata, batch_size=500)

        Notes:
            - For IVF indices (ivf, ivfpq), first batch is used for training if not trained
            - Metadata is stored in self.metadata dictionary with vector indices as keys
            - Batch processing helps manage memory for large datasets
        """ 
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
        Search for k-nearest neighbors of query vector(s) in the index.

        Performs similarity search using the FAISS index and returns distances,
        indices, and associated metadata of the nearest neighbors.

        Args:
            query (np.ndarray): Query vector(s) to search for, shape (n_queries, dimension)
            k (int, optional): Number of nearest neighbors to retrieve. Defaults to 3.
            nprobe (Optional[int], optional): Number of cells to visit for IVF-based indices.
                Only applies to 'ivf' and 'ivfpq' index types. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'distances': Array of distances to nearest neighbors
                - 'indices': Array of indices of nearest neighbors
                - 'metadata': List of metadata for each neighbor

        Example:
            >>> # Search for 5 nearest neighbors of a query vector
            >>> query_vector = np.random.random((1, 128))
            >>> results = search(query_vector, k=5)
            >>> print(results['distances'])  # distances to neighbors
            >>> print(results['metadata'])   # metadata of neighbors
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
        """
        Save the FAISS index, metadata, and configuration to disk.

        Saves all necessary data to reconstruct the index:
        1. FAISS index file
        2. Configuration file with metadata and parameters

        Args:
            path (str): Base path for saving files. Will create:
                - {path}: FAISS index file
                - {path}.config.json: Configuration and metadata

        Notes:
            - If index is on GPU, automatically converts to CPU before saving
            - Configuration includes dimension, index type, PQ config, and metadata
        """
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
        """Load a saved AudioIndex instance from disk.

        Creates a new AudioIndex instance from saved files, optionally
        moving the index to GPU.

        Args:
            path (str): Base path where index was saved
            use_gpu (bool, optional): Whether to move index to GPU. Defaults to True.

        Returns:
            AudioIndex: Loaded index instance with all saved data and configuration

        Example:
            >>> # Load index from disk and move to GPU
            >>> index = AudioIndex.load("path/to/saved/index")
            >>> # Load index and keep on CPU
            >>> index = AudioIndex.load("path/to/saved/index", use_gpu=False)

        Notes:
            - Loads both index file and configuration/metadata
            - Automatically moves index to GPU if requested and available
            - Sets is_trained=True since loaded indices are pre-trained
        """
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