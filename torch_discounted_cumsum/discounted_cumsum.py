import os

import torch
from torch.utils.cpp_extension import load

VERBOSE = False


def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


try:
    import torch_discounted_cumsum_cpu
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_discounted_cumsum_cpu')
    torch_discounted_cumsum_cpu = load(
        name='torch_discounted_cumsum_cpu',
        sources=[
            _resolve('discounted_cumsum_cpu.cpp'),
        ],
        verbose=VERBOSE,
    )


try:
    import torch_discounted_cumsum_cuda
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_discounted_cumsum_cuda')
    torch_discounted_cumsum_cuda = None
    if torch.cuda.is_available():
        torch_discounted_cumsum_cuda = load(
            name='torch_discounted_cumsum_cuda',
            sources=[
                _resolve('discounted_cumsum_cuda.cpp'),
                _resolve('discounted_cumsum_cuda_kernel.cu'),
            ],
            verbose=VERBOSE,
        )


def _discounted_cumsum_left_dispatcher(input, gamma):
    if not torch.is_tensor(input):
        raise ValueError('Input must be a torch.Tensor')
    if not torch.is_tensor(gamma):
        raise ValueError('Gamma must be a torch.Tensor')
    if input.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.discounted_cumsum_left_cuda(input.contiguous(), gamma.contiguous())
    else:
        return torch_discounted_cumsum_cpu.discounted_cumsum_left_cpu(input, gamma)

def _discounted_cumsum3_left_dispatcher(input, gamma):
    if not torch.is_tensor(input):
        raise ValueError('Input must be a torch.Tensor')
    if not torch.is_tensor(gamma):
        raise ValueError('Gamma must be a torch.Tensor')
    if input.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.discounted_cumsum3_left_cuda(input.contiguous(), gamma.contiguous())
    else:
        return torch_discounted_cumsum_cpu.discounted_cumsum3_left_cpu(input, gamma)

def _discounted_cumsum_right_dispatcher(input, gamma):
    if not torch.is_tensor(input):
        raise ValueError('Input must be a torch.Tensor')
    if not torch.is_tensor(gamma):
        raise ValueError('Gamma must be a torch.Tensor')
    if input.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.discounted_cumsum_right_cuda(input.contiguous(), gamma.contiguous())
    else:
        return torch_discounted_cumsum_cpu.discounted_cumsum_right_cpu(input, gamma)

def _discounted_cumsum3_right_dispatcher(input, gamma):
    if not torch.is_tensor(input):
        raise ValueError('Input must be a torch.Tensor')
    if not torch.is_tensor(gamma):
        raise ValueError('Gamma must be a torch.Tensor')
    if input.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.discounted_cumsum3_right_cuda(input.contiguous(), gamma.contiguous())
    else:
        return torch_discounted_cumsum_cpu.discounted_cumsum_right_cpu(input, gamma)

class DiscountedCumSumFunction(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, gamma = inputs
        gamma_requires_grad = gamma.requires_grad
        ctx.save_for_backward(output if gamma_requires_grad else None, gamma)

class DiscountedCumSumLeftFunction(DiscountedCumSumFunction):
    @staticmethod
    def forward(input, gamma):
        output = _discounted_cumsum_left_dispatcher(input.float(), gamma).to(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, gamma = ctx.saved_tensors
        grad_input = _discounted_cumsum_right_dispatcher(grad_output.float(), gamma).to(output)
        grad_gamma = None
        if output is not None:
            z = _discounted_cumsum_left_dispatcher(output, gamma)
            z = z[:, :-1]
            dLdy = grad_output[:, 1:]
            grad_gamma = (z * dLdy).sum(dim=1)
        return grad_input, grad_gamma, None


class DiscountedCumSumRightFunction(DiscountedCumSumFunction):
    @staticmethod
    def forward(input, gamma):
        output = _discounted_cumsum_right_dispatcher(input.float(), gamma).to(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, gamma = ctx.saved_tensors
        grad_input = _discounted_cumsum_left_dispatcher(grad_output.float(), gamma).to(output)
        grad_gamma = None
        if output is not None:
            z = _discounted_cumsum_right_dispatcher(output, gamma)
            z = z[:, 1:]
            dLdy = grad_output[:, :-1]
            #print(z.shape) # (B,S)
            #print(dLdy.shape)
            grad_gamma = (z * dLdy).sum(dim=1)
        return grad_input, grad_gamma, None
  
    # @staticmethod
    # def vmap(info, in_dims, x, gamma):
    #     x_bdim, gamma_bdim = in_dims
    #     # x = x.movedim(x_bdim, 0)
    #     assert x_bdim == 0, "x must be batched over the first dimension"
    #     assert gamma_bdim is None, "gamma must be unbatched"
    #     DiscountedCumSumRightFunction.apply(x, gamma)
    #     # The strategy is: expand {x, ind, ind_inv} to all have the dimension
    #     # being vmapped over.
    #     # Then, call back into NumpyTake(expanded_x, expanded_ind, expanded_ind_inv, new_dim).
    #     return DiscountedCumSumRightFunction.apply(x, gamma), 0


class DiscountedCumSum3RightFunction(DiscountedCumSumFunction):
    @staticmethod
    def forward(input, gamma):
        output = _discounted_cumsum3_right_dispatcher(input.float(), gamma).to(input)
        return output.to(input)

    @staticmethod
    def backward(ctx, grad_output):
        output, gamma = ctx.saved_tensors
        grad_input = _discounted_cumsum3_left_dispatcher(grad_output.float(), gamma).to(output)
        grad_gamma = None
        if output is not None:
            z = _discounted_cumsum3_right_dispatcher(output, gamma)
            z = z[..., 1:] # the head dimension, not the len dim
            dLdy = grad_output[..., :-1]
            # print(z.shape) # (B,H,S)
            # print(dLdy.shape)  # (B,H,S)
            grad_gamma = (z * dLdy).sum(dim=-1)
        return grad_input, grad_gamma, None


class DiscountedCumSum3LeftFunction(DiscountedCumSumFunction):
    @staticmethod
    def forward(input, gamma):
        output = _discounted_cumsum3_left_dispatcher(input.float(), gamma).to(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, gamma = ctx.saved_tensors
        grad_input = _discounted_cumsum3_right_dispatcher(grad_output.float(), gamma).to(output)
        grad_gamma = None
        if output is not None:
            z = _discounted_cumsum3_left_dispatcher(output, gamma)
            z = z[..., :-1]
            dLdy = grad_output[..., 1:]
            grad_gamma = (z * dLdy).sum(dim=-1)
        return grad_input, grad_gamma, None

def discounted_cumsum_left(input, gamma):
    assert torch.is_tensor(input)
    assert torch.is_tensor(gamma)
    if gamma.dim() == 0:
        gamma = gamma.reshape(-1)
    return DiscountedCumSumLeftFunction.apply(input, gamma)

def discounted_cumsum_right(input, gamma):
    assert torch.is_tensor(input)
    assert torch.is_tensor(gamma)
    if gamma.dim() == 0:gamma = gamma.reshape(-1)
    return DiscountedCumSumRightFunction.apply(input, gamma)

def discounted_cumsum3_right(input, gamma):
    assert torch.is_tensor(input)
    assert torch.is_tensor(gamma)
    return DiscountedCumSum3RightFunction.apply(input, gamma)

def discounted_cumsum3_left(input, gamma):
    assert torch.is_tensor(input)
    assert torch.is_tensor(gamma)
    if gamma.dim() == 0:gamma = gamma.reshape(-1)
    return DiscountedCumSum3LeftFunction.apply(input, gamma)