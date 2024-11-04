import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from typing import Optional, Union
from pathlib import Path
import warnings

class AudioFeatureExtractor:
    """Audio feature extraction using wav2vec2."""
    
    def __init__(self, 
                 model_name: str = "facebook/wav2vec2-base",
                 device: Optional[str] = None,
                 target_sr: int = 16000):
        """Initialize feature extractor."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr = target_sr
        
        print(f"Loading wav2vec2 model: {model_name}")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.embedding_dim = self.model.config.hidden_size
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def process_audio(self, 
                     audio_path: Union[str, Path],
                     normalize: bool = True) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
        
        # Normalize if requested
        if normalize:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
        return waveform

    def extract_features(self, 
                        audio: Union[str, Path, torch.Tensor],
                        batch_size: Optional[int] = None) -> np.ndarray:
        """Extract features from audio."""
        if isinstance(audio, (str, Path)):
            audio = self.process_audio(audio)
            
        # Convert to numpy for feature extractor
        audio_np = audio.squeeze().numpy()
        
        # Process in batches if audio is too long
        if batch_size and len(audio_np) > batch_size:
            chunks = np.array_split(audio_np, len(audio_np) // batch_size + 1)
            embeddings = []
            
            for chunk in chunks:
                inputs = self.feature_extractor(
                    chunk,
                    sampling_rate=self.target_sr,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    input_values = inputs.input_values.to(self.device)
                    outputs = self.model(input_values)
                    embedding = torch.mean(outputs.last_hidden_state, dim=1)
                    embeddings.append(embedding.cpu().numpy())
                    
            return np.mean(embeddings, axis=0)
        else:
            inputs = self.feature_extractor(
                audio_np,
                sampling_rate=self.target_sr,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                input_values = inputs.input_values.to(self.device)
                outputs = self.model(input_values)
                embedding = torch.mean(outputs.last_hidden_state, dim=1)
                
            return embedding.cpu().numpy()