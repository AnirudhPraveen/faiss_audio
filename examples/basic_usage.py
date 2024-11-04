import os
import sys
import torch
import gc
import warnings
import random
from pathlib import Path
from faiss_audio import AudioSearch

def debug_print(msg):
    """Print debug message with memory info."""
    print(f"\n{msg}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
    print(f"Step completed at: {msg}")
    sys.stdout.flush()  # Force print

def main():
    try:
        # Suppress warnings
        warnings.filterwarnings('ignore')
        debug_print("Starting script")

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        debug_print("Initial cleanup done")

        # Load test dataset
        dataset_dir = Path("../tests/data")
        all_audio_files = list(dataset_dir.glob("*.wav"))[:10]  # Start with 10 files
        debug_print(f"Found {len(all_audio_files)} files")

        # Randomly select one file for query (hold-out)
        query_file = random.choice(all_audio_files)
        # Remove query file from indexing set
        index_files = [f for f in all_audio_files if f != query_file]
        
        debug_print(f"Selected query file: {query_file.name}")
        debug_print(f"Files for indexing: {len(index_files)}")

        # Initialize search with minimal settings
        debug_print("Initializing AudioSearch")
        search = AudioSearch(
            index_type="flat",  # Simple index type
            use_gpu=False       # CPU only initially
        )
        debug_print("AudioSearch initialized")

        # Index files with metadata
        debug_print("Starting indexing...")
        for i, audio_file in enumerate(index_files):
            debug_print(f"Processing file {i+1}/{len(index_files)}: {audio_file.name}")
            try:
                # Create metadata for the file
                metadata = [{
                    'filename': audio_file.name,
                    'path': str(audio_file),
                }]
                
                # Index with metadata
                search.index_audio_files(
                    str(audio_file),
                    batch_size=1,
                    metadata=metadata
                )
                
                gc.collect()  # Clean up after each file
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing {audio_file.name}: {str(e)}")
                raise

        debug_print("All files processed")

        # Try search with held-out query file
        debug_print(f"Searching with held-out file: {query_file.name}")
        results = search.search(str(query_file), k=3)  # Get top 3 matches

        # Print results
        print("\nSearch Results for query file:", query_file.name)
        print("\nNearest matches from indexed files:")
        for i, match in enumerate(results, 1):
            try:
                print(f"\nMatch {i}:")
                # Add error handling for missing keys
                print(f"File: {match.get('filename', 'Unknown')}")
                print(f"Distance: {match.get('distance', 'Unknown'):.3f}")
                print(f"Path: {match.get('path', 'Unknown')}")
                # Print full match dictionary for debugging
                print("Debug - Full match data:", match)
            except Exception as e:
                print(f"Error printing match {i}: {str(e)}")
                print("Raw match data:", match)

        # Save index and metadata
        debug_print("Saving index...")
        search.save("audio_search.index")
        
        # Memory cleanup
        debug_print("Final cleanup")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        debug_print("Script completed")

if __name__ == "__main__":
    main()