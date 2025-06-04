import torch
import torch.nn.functional as F
from vllm_kernels import _custom_ops # pytype: disable=import-error

from vllm.model_executor.utils import set_weight_attrs 