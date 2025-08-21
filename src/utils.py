
import torch
import gc

def clean_gpu_memory():
    """Aggressively clean GPU memory to prevent OOM errors."""
    # Clear PyTorch's CUDA cache
    torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Force synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()