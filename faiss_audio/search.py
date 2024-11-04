from .features import AudioFeatureExtractor
from .index import AudioIndex
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
from tqdm import tqdm
import warnings

class AudioSearch:
    """High-level audio search functionality with wav2vec2 features."""
    
    def __init__(self,
                 model_name: str = "facebook/wav2vec2-base",
                 index_type: str = "ivfpq",
                 pq_config: Optional[Dict] = None,
                 use_gpu: bool = True):
        """
        Initialize audio search system.
        
        Args:
            model_name: Name of wav2vec2 model to use
            index_type: Type of FAISS index
            pq_config: Configuration for Product Quantization
            use_gpu: Whether to use GPU
        """
        # Initialize feature extractor
        self.feature_extractor = AudioFeatureExtractor(
            model_name=model_name,
            device="cuda" if use_gpu else "cpu"
        )
        
        # Initialize index
        self.index = AudioIndex(
            dimension=self.feature_extractor.embedding_dim,
            index_type=index_type,
            pq_config=pq_config,
            use_gpu=use_gpu
        )

    def index_audio_files(self, 
                         audio_paths: Union[str, List[str], Path],
                         batch_size: int = 32,
                         show_progress: bool = True) -> None:
        """
        Index audio files.
        
        Args:
            audio_paths: Single path or list of paths to audio files
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
        """
        # Handle directory input
        if isinstance(audio_paths, (str, Path)):
            path = Path(audio_paths)
            if path.is_dir():
                audio_paths = list(path.glob("**/*.wav")) + list(path.glob("**/*.mp3"))
            else:
                audio_paths = [path]

        embeddings = []
        metadata = []
        
        # Process files
        iterator = tqdm(audio_paths) if show_progress else audio_paths
        
        for audio_path in iterator:
            try:
                # Extract features
                embedding = self.feature_extractor.extract_features(audio_path)
                
                # Store data
                embeddings.append(embedding)
                metadata.append({
                    'path': str(audio_path),
                    'filename': Path(audio_path).name
                })
                
                # Add to index in batches
                if len(embeddings) >= batch_size:
                    self.index.add(np.vstack(embeddings), metadata)
                    embeddings = []
                    metadata = []
                    
            except Exception as e:
                warnings.warn(f"Error processing {audio_path}: {str(e)}")
                continue
        
        # Add remaining files
        if embeddings:
            self.index.add(np.vstack(embeddings), metadata)

    def search(self, 
               query: Union[str, Path, np.ndarray],
               k: int = 5,
               nprobe: int = 10) -> List[Dict]:
        """
        Search for similar audio.
        
        Args:
            query: Audio file path or pre-computed embedding
            k: Number of results to return
            nprobe: Number of clusters to visit
        """
        # Get query embedding
        if isinstance(query, (str, Path)):
            query_embedding = self.feature_extractor.extract_features(query)
        else:
            query_embedding = query
            
        # Search
        results = self.index.search(query_embedding, k=k, nprobe=nprobe)
        
        # Format results
        matches = []
        for i in range(k):
            matches.append({
                'distance': float(results['distances'][0][i]),
                **results['metadata'][0][i]
            })
            
        return matches

    def save(self, path: str) -> None:
        """Save the search system."""
        self.index.save(path)
        
    @classmethod
    def load(cls, 
             path: str,
             model_name: str = "facebook/wav2vec2-base",
             use_gpu: bool = True) -> 'AudioSearch':
        """Load a saved search system."""
        search = cls(model_name=model_name, use_gpu=use_gpu)
        search.index = AudioIndex.load(path, use_gpu=use_gpu)
        return search