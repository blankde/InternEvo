# adopted from https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm

import numbers
import importlib

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.utils.logger import get_logger

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()

# try:
from apex.normalization.fused_layer_norm import mixed_dtype_fused_rms_norm_affine
from apex._autocast_utils import _cast_if_autocast_enabled

apex_rmsnorm_impl = True
# except (ModuleNotFoundError, ImportError):
#     logger.warning("The torch implementation for MixFusedRMSNorm is slower than apex. Please note this!")
#     apex_rmsnorm_impl = False

try:
    from deeplink_ext.internevo_ops import MixedFusedRMSNorm as _RMSNormDIPU

    deeplink_rmsnorm_impl = True
except (ModuleNotFoundError, ImportError):
    deeplink_rmsnorm_impl = False

try:
    from torch_npu import npu_rms_norm

    torchnpu_rmsnorm_impl = True
except (ModuleNotFoundError, ImportError):
    torchnpu_rmsnorm_impl = False


def manual_rms_norm(my_input, weight, normalized_shape, eps, add_unit_offset=False):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = my_input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    my_input = my_input * torch.rsqrt(variance + eps)

    if weight is None:
        return my_input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        my_input = my_input.to(weight.dtype)

    if add_unit_offset:
        return (1 + weight) * my_input
    else:
        return weight * my_input

global fused_layer_norm_cuda
fused_layer_norm_cuda = None

class FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps, memory_efficient=False):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.memory_efficient = memory_efficient
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_forward_affine(
            input_, ctx.normalized_shape, weight_, ctx.eps)
        if ctx.memory_efficient:
            ctx.save_for_backward(output, weight_, invvar)
        else:
            ctx.save_for_backward(input_, weight_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_or_output, weight_, invvar = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_input, grad_weight = fused_layer_norm_cuda.rms_backward_affine(
           grad_output.contiguous(), invvar, input_or_output,
           ctx.normalized_shape, weight_, ctx.eps, ctx.memory_efficient
        )
        return grad_input, grad_weight, None, None, None

class FusedRMSNormAffineMixedDtypesFunction(FusedRMSNormAffineFunction):

    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps, memory_efficient=False):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.memory_efficient = memory_efficient
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_forward_affine_mixed_dtypes(
            input_, ctx.normalized_shape, weight_, ctx.eps
        )
        if ctx.memory_efficient:
            ctx.save_for_backward(output, weight_, invvar)
        else:
            ctx.save_for_backward(input_, weight_, invvar)
        return output

class _RMSNorm(torch.nn.Module):
    """A generic module for RMS normalization."""

    def __init__(self, normalized_shape, eps=1e-5, add_unit_offset=False):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.empty(*normalized_shape))
        self.add_unit_offset = add_unit_offset
        self.reset_parameters()

    def forward(self, _input: torch.Tensor):
        if apex_rmsnorm_impl:
            _norm_func = mixed_dtype_fused_rms_norm_affine
            return _norm_func(_input, self.weight, self.normalized_shape, self.eps)
        else:
            _norm_func = manual_rms_norm
            return _norm_func(_input, self.weight, self.normalized_shape, self.eps, self.add_unit_offset)


    def explicit_fwd(self, ctx, _input: torch.Tensor):
        if apex_rmsnorm_impl:
            args = _cast_if_autocast_enabled(_input, self.weight, self.normalized_shape, self.eps)
            with torch.amp.autocast('cuda', enabled=False):
                return FusedRMSNormAffineMixedDtypesFunction.forward(ctx, *args)
        else:
            assert False

    def explicit_bwd(self, ctx, grad_output: torch.Tensor):
        if apex_rmsnorm_impl:
            with torch.amp.autocast('cuda', enabled=False):
                grad_input, grad_weight, *_ = FusedRMSNormAffineMixedDtypesFunction.backward(ctx, grad_output)
        else:
            assert False   

        return grad_input, grad_weight

    def reset_parameters(self):
        if self.add_unit_offset:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, "


class _RMSNormNPU(torch.nn.Module):
    """A custom NPU module for RMS normalization."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.empty(*normalized_shape))
        self.reset_parameters()
        self.rmsorm_npu_forward = npu_rms_norm

    def forward(self, _input: torch.Tensor):
        weight_fp32 = self.weight.to(torch.float32)
        input_fp32 = _input.to(torch.float32)
        output = self.rmsorm_npu_forward(input_fp32, gamma=weight_fp32, epsilon=self.eps)[0].to(self.weight.dtype)
        return output

    def reset_parameters(self):
        init.ones_(self.weight)

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, ".format(**self.__dict__)


# TODO: Support deeplink in a more unified manner
backend = internlm_accelerator.get_accelerator_backend()
if backend in [AcceleratorType.DIPU, AcceleratorType.DITORCH] and deeplink_rmsnorm_impl:
    RMSNorm = _RMSNormDIPU
elif backend == AcceleratorType.NPU and torchnpu_rmsnorm_impl:
    RMSNorm = _RMSNormNPU
else:
    RMSNorm = _RMSNorm
