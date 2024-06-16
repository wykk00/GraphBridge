import torch.distributed as dist
import os
import torch

# Check distributed environment
def is_dist():
    return False if os.getenv("WORLD_SIZE") is None or eval(os.getenv("WORLD_SIZE")) <= 1 else True

# Initialize distributed environment
def set_dist_env():
    dist.init_process_group('nccl', init_method='env://')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    return rank