import faiss
import torch
import numpy as np
from typing import Optional, Dict

def get_gpu_resources() -> dict:
    """Get available GPU resources for FAISS."""
    try:
        ngpus = faiss.get_num_gpus()
        gpu_resources = {}
        if ngpus > 0:
            for i in range(ngpus):
                res = faiss.StandardGpuResources()
                res.setTempMemory(1024 * 1024 * 512)  # 512MB temp memory
                gpu_resources[i] = res
        return gpu_resources
    except AttributeError:
        return {}

def index_cpu_to_gpu(index: faiss.Index, 
                     gpu_id: Optional[int] = None, 
                     resources: Optional[dict] = None) -> faiss.Index:
    """Convert a CPU index to GPU."""
    if not resources:
        return index

    if gpu_id is None:
        gpu_id = 0
        try:
            free_memory = []
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                free_memory.append(torch.cuda.memory_allocated())
            gpu_id = free_memory.index(min(free_memory))
        except:
            pass

    try:
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(resources[gpu_id], gpu_id, index, co)
        return gpu_index
    except Exception as e:
        print(f"Warning: GPU conversion failed - {str(e)}")
        return index