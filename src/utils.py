
import torch
import gc

def clean_gpu_memory():
    """Aggressively clean GPU/MPS memory to prevent OOM errors."""
    # Force garbage collection first
    gc.collect()
    
    # Clear device caches based on availability
    if torch.cuda.is_available():
        # Clear PyTorch's CUDA cache
        torch.cuda.empty_cache()
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        # Force synchronization
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        # Clear MPS cache on Mac M series
        torch.mps.empty_cache()
        # Force synchronization
        torch.mps.synchronize()