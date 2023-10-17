import os

import torch
from torch.utils.cpp_extension import load
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
VERBOSE = False


def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)



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

def _weighted_cumsum_dispatcher(input, weight):
    if not torch.is_tensor(input):
        raise ValueError('Input must be a torch.Tensor')
    if not torch.is_tensor(weight):
        raise ValueError('weight must be a torch.Tensor')
    if input.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.weighted_cumsum_cuda(input.contiguous(), weight.contiguous())
    else:
        raise 

def _weighted_cumsum_batch_dispatcher(input, weight):
    if not torch.is_tensor(input):
        raise ValueError('Input must be a torch.Tensor')
    if not torch.is_tensor(weight):
        raise ValueError('weight must be a torch.Tensor')
    if input.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.weighted_cumsum_batch_cuda(input.contiguous(), weight.contiguous())
    else:
        raise 


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

class WeightedCumSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, weight):
        output = _weighted_cumsum_dispatcher(input.float(), weight).to(input)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight = inputs
        weight_requires_grad = weight.requires_grad
        ctx.save_for_backward(output if weight_requires_grad else None, weight)

    @staticmethod
    def backward(ctx, grad_output):
        output, weight = ctx.saved_tensors
        grad_input = _weighted_cumsum_left_dispatcher(grad_output.float(), weight).to(output)
        grad_gamma = None
        if output is not None:
            raise 
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

class WeightedCumSumBatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, weight):
        output = _weighted_cumsum_batch_dispatcher(input.float(), weight).to(input)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight = inputs
        weight_requires_grad = weight.requires_grad
        ctx.save_for_backward(output if weight_requires_grad else None, weight)

    @staticmethod
    def backward(ctx, grad_output):
        output, weight = ctx.saved_tensors
        grad_input = _weighted_cumsum_left_dispatcher(grad_output.float(), weight).to(output)
        grad_gamma = None
        if output is not None:
            raise 
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

def weighted_cumsum(input, weight):
    assert torch.is_tensor(input)
    assert torch.is_tensor(weight)
    return WeightedCumSumFunction.apply(input, weight)

def weighted_cumsum_batch(input, weight):
    assert torch.is_tensor(input)
    assert torch.is_tensor(weight)
    return WeightedCumSumBatchFunction.apply(input, weight)

def _qkvg_retention_dispatcher(q, k,v,g):
    if not torch.is_tensor(q):raise ValueError('q must be a torch.Tensor')
    if not torch.is_tensor(k):raise ValueError('k must be a torch.Tensor')
    if not torch.is_tensor(v):raise ValueError('v must be a torch.Tensor')
    if not torch.is_tensor(g):raise ValueError('g must be a torch.Tensor')

    if q.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.qkvg_retention_cuda(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous())
    else:
        raise 

class QKVGFunction(torch.autograd.Function):
    @staticmethod
    def forward(q, k,v,g):
        output = _qkvg_retention_dispatcher(q.float(), k.float(), v.float(), g.float()).to(q)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k,v,g = inputs
        weight_requires_grad = g.requires_grad
        ctx.save_for_backward(output if weight_requires_grad else None, g)

    @staticmethod
    def backward(ctx, grad_output):
        raise 

def qkvg_retention(q, k,v,g):
    return QKVGFunction.apply(q, k,v,g)




def get_omask(slen,num_heads):
    decay = torch.log(1 - 2**(-5 - torch.arange(num_heads, dtype=torch.float)))
    index= torch.arange(slen).float()
    mask = torch.tril(torch.ones(slen, slen))
    mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), torch.inf)
    mask = torch.exp(mask * decay[:, None, None])

    mask = torch.nan_to_num(mask)
    #omask = mask.unsqueeze(0)  # [1, h, t, t]
    # mask = omask / omask.sum(dim=-1, keepdim=True).sqrt()
    # mask = torch.nan_to_num(mask, nan=0.0)
    return mask

if __name__ == '__main__':
    B=1
    D=1
    H=1
    S=5
    # q = torch.randn(B, H, S, D).cuda()
    # k = torch.randn(B, H, S, D).cuda()
    # v = torch.randn(B, H, S, D).cuda()
    # g = get_omask(S,H).cuda()
    # o1 = qkvg_retention(q,k,v,g)
    # o2 = torch.einsum('bhia,bhja,bhjc,hij->bhic', q, k, v, g)
    # print(torch.dist(o1,o2))
    # print("="*10)
    # print(q[0,0])
    # print(k[0,0])
    # print(v[0,0])
    # print(g[0])
    # print("="*10)
    # print(o1[0,0])
    # print(o2[0,0])
    # exit()







    # K = 2
    # gamma = torch.Tensor([0.99]).cuda()
    # x = torch.ones(1, N).cuda()
    # y_N = discounted_cumsum_right(x, gamma)
    # print(y_N)

    # gamma = torch.Tensor([0.99, 0.98, 0.07]).cuda()
    # x   = torch.ones(3, N).cuda()
    # y_N = discounted_cumsum_right(x, gamma)
    # print(y_N)
    # print("="*20)

    

    #gamma = torch.Tensor([0.99, 0.98, 0.07]).cuda()

    omask = get_omask(S,H).cuda() # (H,S,S) -> 
    gamma = omask[:,1,0]
    x     = torch.randn(B,H,S).cuda() 
    #y_N1 = discounted_cumsum_left(x.flatten(0,1), gamma.repeat(B)).reshape(B,H,S)
    y_N1 = discounted_cumsum_left(x[:,0], gamma[0].repeat(B)) #(B,S) and (H)
    #print(y_N1.mean())
    y_N3 = weighted_cumsum(x[:,0],omask[0])
    #print(y_N3.mean())
    #print(x[:,0].shape)
    #print(omask[0].shape)
    print(torch.dist(y_N1,y_N3))
    #print(y_N1.shape)
    #print(y_N1)
    print("==============================")
    y_N1 = discounted_cumsum_left(x.flatten(0,1), gamma.repeat(B)).reshape(B,H,S)
    y_N2 = discounted_cumsum3_left(x,gamma)
    y_N3 = weighted_cumsum_batch(x,omask)
    print(x.shape)
    print(gamma.shape)
    print(omask.shape)
    print(torch.dist(y_N1,y_N3))
    print(torch.dist(y_N1,y_N2))
    #print(y_N1.shape)
    #print(y_N1)
    # print(gamma)
    # print(x)
    # print(omask[0])
    # print("==============================")
    # print(y_N1)
    # print(y_N3)
    # print(y_N1[0])
    # print(y_N1[-1])
    # print("="*20)
    #y_N2 = discounted_cumsum_right(x.flatten(0,2), gamma.repeat(B*D))
   
    #y_N3 = weighted_cumsum_right(, )
    exit()
    # print(y_N2[0])
    # print(y_N2[-1])
    # print("=============")
    print(y_N1.shape)
    for i, (a1, a2) in enumerate(zip(y_N1,y_N2)):
        print(f"{i:3d} => {torch.dist(a1,a2):.4f} => {a1} <-> {a2}")
        #print(f"{i:3d} => {torch.dist(a1,a2):.4f}")
    #print(torch.dist(y_N1,y_N2))

    ################################################################################
    ######## VMAP does not work since the custom vamp rule is tricky so far ########
    ################################################################################
    # batched_discounted_cumsum = torch.vmap(discounted_cumsum_right, in_dims=(0, None))
    # print(x.shape)
    # y_N3 = batched_discounted_cumsum(x.flatten(0,1),gamma)
    # ==============================================================================



    # B=2
    # D=16
    # H=16
    # S=30
    # Output= []
    # for i in range(2):
    #     model  = torch.nn.ModuleList([torch.nn.Conv1d(H,H,3,1,1,bias=False) for _ in range(2)])
    #     # x = torch.randn(B, H, S).cuda()
    #     # torch.save(model.state_dict(),'test.weight.pt')
    #     #torch.save({"x":x,'gamma':gamma},'test.input.pt' )
    #     model.load_state_dict(torch.load('test.weight.pt'))
    #     model = model.cuda()
    #     x = torch.load('test.input.pt')['x'].cuda()
    #     gamma  = torch.load('test.input.pt')['gamma'].cuda() #torch.rand(H).cuda()/20+0.945
    #     gamma.requires_grad=True
    #     if i == 0:
    #         x = discounted_cumsum_right(model[0](x).flatten(0,1), gamma.repeat(B)).reshape(B,H,S)
    #         x = discounted_cumsum_right(model[1](x).flatten(0,1), gamma.repeat(B)).reshape(B,H,S)
    #     elif i==1:
    #         x = discounted_cumsum3_right(model[0](x), gamma)
    #         x = discounted_cumsum3_right(model[1](x), gamma)
    #     else:
    #         raise NotImplementedError
    #     #print(y_N[1])
    #     loss = x.sum()
    #     loss.backward()
    #     Output.append(gamma.grad)
    # print(Output[0])
    # print(Output[1])

    # print(gamma.grad)
    # # tensor = torch.randn(B, D, D  ,H, S).cuda()
    # # gamma  = torch.randn(H).cuda()
    # # print(tensor.shape)
    # # print(gamma.shape)
    # # y_N = discounted_cumsum3_right(tensor.flatten(0, 2), gamma)
    # # print(y_N[1])
