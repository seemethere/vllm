# This file makes vllm_kernels a Python package 

# Expose submodules for now. Specific functions/classes can be re-exported later
# as the API for vllm-kernels is solidified.
from . import allocator
from . import custom_ops
from . import flash_attn

# Example of how specific symbols might be exposed in the future:
# from .custom_ops import paged_attention_v1, rotary_embedding
# from .allocator import VllmAllocator

# __all__ can be defined later to specify the public API
# __all__ = ["allocator", "custom_ops", "flash_attn", "paged_attention_v1", ...] 