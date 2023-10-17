#include <torch/extension.h>


enum SumDirection {
    SUM_DIRECTION_LEFT,
    SUM_DIRECTION_RIGHT,
};


template <SumDirection sum_direction>
__device__ __forceinline__
void resolve_positions(
    const int &stride_prev_group, const int &stride_cur_group, const int &group_of_thread, const int &thread_in_group,
    int &change_pos, int &discounted_pos, int &discount_power
);


template <>
__device__ __forceinline__
void resolve_positions<SUM_DIRECTION_LEFT>(
    const int &stride_prev_group, const int &stride_cur_group, const int &group_of_thread, const int &thread_in_group,
    int &change_pos, int &discounted_pos, int &discount_power
) {
    change_pos = group_of_thread * stride_cur_group + thread_in_group + stride_prev_group;
    discounted_pos = group_of_thread * stride_cur_group + stride_prev_group - 1;
    discount_power = thread_in_group + 1;
}


template <>
__device__ __forceinline__
void resolve_positions<SUM_DIRECTION_RIGHT>(
    const int &stride_prev_group, const int &stride_cur_group, const int &group_of_thread, const int &thread_in_group,
    int &change_pos, int &discounted_pos, int &discount_power
) {
    change_pos = group_of_thread * stride_cur_group + thread_in_group;
    discounted_pos = group_of_thread * stride_cur_group + stride_prev_group;
    discount_power = stride_prev_group - thread_in_group;
}


template <typename scalar_t>
__device__ __forceinline__
scalar_t discounted_sum_power(scalar_t a, scalar_t b, scalar_t gamma, int power) {
    return a + b * pow(gamma, scalar_t(power));
}



int getOptimalThreadsPerBlock(int problemSize, int dim, int max) {
    // Get device properties
    cudaDeviceProp deviceProp;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    int maxThreadsPerBlock = deviceProp.maxThreadsDim[dim];
    
    if (problemSize < maxThreadsPerBlock) {
        if (problemSize < max) {
            return problemSize;
        } else {
            return max;
        }
    } else {
        if (maxThreadsPerBlock < max) {
            return maxThreadsPerBlock;
        } else {
            return max;
        }
    }
}


template <typename scalar_t, SumDirection sum_direction>
__global__
void discounted_cumsum_kernel_stage(
    torch::PackedTensorAccessor32<scalar_t, 2> x,
    torch::PackedTensorAccessor32<scalar_t, 1> gamma,
    int stage,
    bool gamma_scalar
) {
    const int len = x.size(1);
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (thread_idy >= x.size(0)) {
        return;
    }

    int stride_prev_group = 1 << stage;
    int stride_cur_group = stride_prev_group << 1;

    int group_of_thread = thread_idx >> stage;
    int thread_in_group = thread_idx - (group_of_thread << stage);

    int change_pos, discounted_pos, discount_power;
    resolve_positions<sum_direction>(
        stride_prev_group, stride_cur_group, group_of_thread, thread_in_group,
        change_pos, discounted_pos, discount_power
    );

    if (change_pos >= len || discounted_pos >= len) {
        return;
    }

    scalar_t gamma_item = gamma_scalar ? gamma[0] : gamma[thread_idy];

    x[thread_idy][change_pos] = discounted_sum_power(
        x[thread_idy][change_pos],
        x[thread_idy][discounted_pos],
        gamma_item,
        discount_power
    );
}











inline
int log2ceil(int x) {
    return (int)ceil(log2((float)x));
}


template <SumDirection sum_direction>
torch::Tensor discounted_cumsum(torch::Tensor x, torch::Tensor gamma) {
    // Minimum required number of threads, assigns them dynamically to respective positions upon each iteration.
    // Results in uncoalesced writes, which is still faster than coalesced writes with half threads idling.

    TORCH_CHECK(x.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input must be 2-dimensional");
    TORCH_CHECK(gamma.device().is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(gamma.dim() == 1, "Gamma must be 1-dimensional");
    TORCH_CHECK(gamma.size(0) == 1 || gamma.size(0) == x.size(0), "Gamma dimensions must be compatible with the input");
    TORCH_CHECK(x.dtype() == gamma.dtype(), "Argument data types must match");

    bool gamma_scalar = gamma.size(0) != x.size(0);

    if (x.size(1) == 0) {
        return x;
    }

    auto y = x.clone();

    const int H = x.size(0);
    const int S = x.size(1);
    const int nstages = log2ceil(S);
    const int St = 1 << (nstages - 1);
    
    //const dim3 blocks((threads_total_x + threads - 1) / threads, x.size(1), x.size(0));
    const int threads_S = getOptimalThreadsPerBlock(St, 0, 1024); //only need half of the dimension. In each dimension, we fully fill O(S) computation
    const int threads_H = getOptimalThreadsPerBlock( H, 1, 1024/threads_S);
    // thread has limit per each channel
    // and totally thread for one block is limited 1024
    const dim3 threads(threads_S, threads_H);
    const dim3 blocks((St + threads_S - 1)/ threads_S, 
                      (H + threads_H - 1) / threads_H);

    // const dim3 blocks((threads_total_x + threads - 1) / threads, x.size(0));

    for (int stage=0; stage<nstages; stage++) {
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "discounted_cumsum_kernel_stage", ([&] {
            discounted_cumsum_kernel_stage<scalar_t, sum_direction><<<blocks, threads>>>(
                y.packed_accessor32<scalar_t, 2>(),
                gamma.packed_accessor32<scalar_t, 1>(),
                stage,
                gamma_scalar
            );
        }));
    }

    return y;
}

// ...

template <typename scalar_t, SumDirection sum_direction>
__global__
void discounted_cumsum_kernel_stage3(
    torch::PackedTensorAccessor32<scalar_t, 3> x,
    torch::PackedTensorAccessor32<scalar_t, 1> gamma,
    int stage,
    bool gamma_scalar
) {
    const int len = x.size(2);
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (thread_idy >= x.size(1) || thread_idz >= x.size(0) ) {
        return;
    }

    int stride_prev_group = 1 << stage;
    int stride_cur_group = stride_prev_group << 1;

    int group_of_thread = thread_idx >> stage;
    int thread_in_group = thread_idx - (group_of_thread << stage);

    int change_pos, discounted_pos, discount_power;
    resolve_positions<sum_direction>(
        stride_prev_group, stride_cur_group, group_of_thread, thread_in_group,
        change_pos, discounted_pos, discount_power
    );

    if (change_pos >= len || discounted_pos >= len) {
        return;
    }

    scalar_t gamma_item = gamma_scalar ? gamma[0] : gamma[thread_idy];

    x[thread_idz][thread_idy][change_pos] = discounted_sum_power(
        x[thread_idz][thread_idy][change_pos],
        x[thread_idz][thread_idy][discounted_pos],
        gamma_item,
        discount_power
    );
}


// template <typename scalar_t, SumDirection sum_direction>
// __global__
// void discounted_cumsum_kernel_contract(
//     torch::PackedTensorAccessor32<scalar_t, 3> q,
//     torch::PackedTensorAccessor32<scalar_t, 3> k,
//     torch::PackedTensorAccessor32<scalar_t, 3> v,
//     torch::PackedTensorAccessor32<scalar_t, 1> g,
//     int stage,
//     bool gamma_scalar,
// ) {
//     const int len = q.size(2);
//     const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int thread_idy = blockIdx.y * blockDim.y + threadIdx.y;
//     const int thread_idz = blockIdx.z * blockDim.z + threadIdx.z;

//     if (thread_idy >= x.size(1) || thread_idz >= x.size(0)  ) {
//         return;
//     }

//     int stride_prev_group = 1 << stage;
//     int stride_cur_group = stride_prev_group << 1;

//     int group_of_thread = thread_idx >> stage;
//     int thread_in_group = thread_idx - (group_of_thread << stage);

//     int change_pos, discounted_pos, discount_power;
//     resolve_positions<sum_direction>(
//         stride_prev_group, stride_cur_group, group_of_thread, thread_in_group,
//         change_pos, discounted_pos, discount_power
//     );

//     if (change_pos >= len || discounted_pos >= len) {
//         return;
//     }

//     scalar_t gamma_item = gamma_scalar ? gamma[0] : gamma[thread_idy];

//     x[thread_idz][thread_idy][change_pos] = discounted_sum_power(
//         x[thread_idz][thread_idy][change_pos],
//         x[thread_idz][thread_idy][discounted_pos],
//         gamma_item,
//         discount_power
//     );
// }

// ...

template <SumDirection sum_direction>
torch::Tensor discounted_cumsum3(torch::Tensor x, torch::Tensor gamma) {
    // ...

    TORCH_CHECK(x.dim() == 3, "Input must be 4-dimensional");
    TORCH_CHECK(gamma.dim() == 1, "Gamma must be 1-dimensional");
    TORCH_CHECK(gamma.size(0) == x.size(1), "Gamma dimensions must be compatible with the input");

    if (x.size(2) == 0) {
        return x;
    }
    const int B = x.size(0);
    const int H = x.size(1);
    const int S = x.size(2);

    auto y = x.clone();

    //const int threads = 64;
    const int nstages = log2ceil(x.size(2));
    const int St = 1 << (nstages - 1);
    //const dim3 blocks((threads_total_x + threads - 1) / threads, x.size(1), x.size(0));
    const int threads_S = getOptimalThreadsPerBlock(St, 0, 1024); //only need half of the dimension. In each dimension, we fully fill O(S) computation
    const int threads_H = getOptimalThreadsPerBlock( H, 1, 1024/threads_S);
    const int threads_B = getOptimalThreadsPerBlock( B, 2, 1024/threads_S/threads_H);
    // thread has limit per each channel
    // and totally thread for one block is limited 1024
    const dim3 threads(threads_S, threads_H, threads_B);
    const dim3 blocks((St + threads_S - 1)/ threads_S, 
                      (H + threads_H - 1) / threads_H,
                      (B + threads_B - 1) / threads_B);

    for (int stage=0; stage<nstages; stage++) {
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "discounted_cumsum_kernel_stage", ([&] {
            discounted_cumsum_kernel_stage3<scalar_t, sum_direction><<<blocks, threads>>>(
                y.packed_accessor32<scalar_t, 3>(),
                gamma.packed_accessor32<scalar_t, 1>(),
                stage,
                false
            );
        }));
    }

    return y;
}


//discounted_cumsum_contract is meanless, since we can not avoid genereate the intermediate
// template <SumDirection sum_direction>
// torch::Tensor discounted_cumsum_contract(torch::Tensor q, //(BH,D1,S)
//                                          torch::Tensor k, //(BH,D1,S)
//                                          torch::Tensor v, //(BH,D2,S)
//                                          torch::Tensor g  //(BH,)
// ) {
//     // ...

//     TORCH_CHECK(q.device().is_cuda(), "Input q must be a CUDA tensor");
//     TORCH_CHECK(k.device().is_cuda(), "Input k must be a CUDA tensor");
//     TORCH_CHECK(v.device().is_cuda(), "Input v must be a CUDA tensor");
//     TORCH_CHECK(g.device().is_cuda(), "Gamma g must be a CUDA tensor");

//     TORCH_CHECK(q.is_contiguous()   , "Input q must be contiguous");
//     TORCH_CHECK(k.is_contiguous()   , "Input k must be contiguous");
//     TORCH_CHECK(v.is_contiguous()   , "Input v must be contiguous");

//     TORCH_CHECK(q.dim() == 3, "Input q must be 3-dimensional");
//     TORCH_CHECK(k.dim() == 3, "Input k must be 3-dimensional");
//     TORCH_CHECK(v.dim() == 3, "Input v must be 3-dimensional");
//     TORCH_CHECK(g.dim() == 1, "Gamma g must be 1-dimensional");

//     TORCH_CHECK(x.dtype() == y.dtype(), "Argument data x-y types must match");
//     TORCH_CHECK(x.dtype() == z.dtype(), "Argument data x-z types must match");
//     TORCH_CHECK(x.dtype() == g.dtype(), "Argument data x-g types must match");

//     // bool gamma_scalar = g.size(0) != x.size(0);
//     TORCH_CHECK(g.size(0) == x.size(0), "Batchsize level must match");
//     TORCH_CHECK(q.size(1) == k.size(1), "QK dimension  must match");
//     TORCH_CHECK(k.size(2) == v.size(2), "KV sequence length  must match");
//     TORCH_CHECK(q.size(2) == v.size(2), "QV sequence length  must match");
    
//     int D1=q.size(1);
//     int D2=v.size(1);
//     int S1=v.size(2);   
    
//     const int threads = 64;
//     const int nstages = log2ceil(S2);
//     const int threads_total_x = 1 << (nstages - 1);
    
//     const dim3 blocks((threads_total_x + threads - 1) / threads, x.size(0));

//     auto o = v.clone();

//     for (int stage=0; stage<nstages; stage++) {
//         AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "discounted_cumsum_kernel_contract", ([&] {
//             discounted_cumsum_kernel_contract<scalar_t, sum_direction><<<blocks, threads>>>(
//                 q.packed_accessor32<scalar_t, 3>(),
//                 k.packed_accessor32<scalar_t, 3>(),
//                 v.packed_accessor32<scalar_t, 3>(),
//                 g.packed_accessor32<scalar_t, 1>(),
//                 stage,
//                 False,
//                 o.packed_accessor32<scalar_t, 3>(),
//             );
//         }));
//     }

//     return p;
// }


// ...

torch::Tensor discounted_cumsum3_left_cuda(torch::Tensor x, torch::Tensor gamma) {
    return discounted_cumsum3<SUM_DIRECTION_LEFT>(x, gamma);
}


torch::Tensor discounted_cumsum3_right_cuda(torch::Tensor x, torch::Tensor gamma) {
    return discounted_cumsum3<SUM_DIRECTION_RIGHT>(x, gamma);
}

torch::Tensor discounted_cumsum_left_cuda(torch::Tensor x, torch::Tensor gamma) {
    return discounted_cumsum<SUM_DIRECTION_LEFT>(x, gamma);
}


torch::Tensor discounted_cumsum_right_cuda(torch::Tensor x, torch::Tensor gamma) {
    return discounted_cumsum<SUM_DIRECTION_RIGHT>(x, gamma);
}





template <typename scalar_t>
__global__
void weighted_cumsum_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> y,       //  (B, S)
    torch::PackedTensorAccessor32<scalar_t, 2> x,       //  (B, S)
    torch::PackedTensorAccessor32<scalar_t, 2> weight  //  (S, S)
) {
    const int len = x.size(1);
    int lt;
    if (len % 2 == 0) {
        lt = len / 2;
    } else {
        lt = (len - 1) / 2 + 1;
    }

    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (thread_idy >= x.size(0) || thread_idx >= lt ) {
        return;
    }
    const int left_pos = thread_idx;
    y[thread_idy][left_pos] = 0;
    for (int i = 0; i <=left_pos; i++) {
        y[thread_idy][left_pos] += x[thread_idy][i] * weight[left_pos][i];
    }
    const int right_pos = len - thread_idx - 1;
    y[thread_idy][right_pos] = 0;
    for (int i = 0; i <=right_pos; i++) {
        y[thread_idy][right_pos] += x[thread_idy][i] * weight[right_pos][i];
    }
}


torch::Tensor weighted_cumsum(torch::Tensor x, torch::Tensor w) {
    // Minimum required number of threads, assigns them dynamically to respective positions upon each iteration.
    // Results in uncoalesced writes, which is still faster than coalesced writes with half threads idling.

    TORCH_CHECK(x.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input must be 2-dimensional");

    TORCH_CHECK(w.device().is_cuda(),   "weight must be a CUDA tensor");
    TORCH_CHECK(w.dim() == 2,           "weight must be 1-dimensional");
    TORCH_CHECK(w.size(1) == x.size(1), "weight dimensions must be compatible with the input");
    TORCH_CHECK(w.size(0) == w.size(1), "weight must be a square matrix");

    if (x.size(1) == 0) {
        return x;
    }

    //int maxThreadsDimZ = deviceProp.maxThreadsDim[2];


    auto y = x.clone();

    const int B = x.size(0);
    const int S = x.size(1);
    
    
    int St;
    if (S % 2 == 0) {
        St = S / 2;
    } else {
        St = (S - 1) / 2 + 1;
    }
    const int threads_S = getOptimalThreadsPerBlock(St, 0, 512 ); //only need half of the dimension. In each dimension, we fully fill O(S) computation
    const int threads_B = getOptimalThreadsPerBlock(B, 1, 1024/threads_S);

    const dim3 threads(threads_S, threads_B);
    const dim3 blocks((St + threads_S - 1)/ threads_S, (B + threads_B - 1) / threads_B);
    // printf("Threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    // printf("Blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
    
    // cudaDeviceProp deviceProp;
    // int device;
    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&deviceProp, device);
    // printf("MaxThreads: (%d, %d, %d),(%d)\n", deviceProp.maxThreadsDim[0], 
    //                                      deviceProp.maxThreadsDim[1], 
    //                                      deviceProp.maxThreadsDim[2],
    //                                      deviceProp.maxThreadsPerBlock);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "weighted_cumsum_kernel", ([&] {
        weighted_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            y.packed_accessor32<scalar_t, 2>(),
            x.packed_accessor32<scalar_t, 2>(),
            w.packed_accessor32<scalar_t, 2>()
        );
    }));
    

    return y;
}

torch::Tensor weighted_cumsum_cuda(torch::Tensor x, torch::Tensor w) {
    return weighted_cumsum(x, w);
}


template <typename scalar_t>
__global__
void weighted_cumsum_kernel_batch(
    torch::PackedTensorAccessor32<scalar_t, 3> y,       //  (B, H, S)
    torch::PackedTensorAccessor32<scalar_t, 3> x,       //  (B, H, S)
    torch::PackedTensorAccessor32<scalar_t, 3> weight  //  (H, S, S)
) {
    const int len = x.size(2);
    int lt;
    if (len % 2 == 0) {
        lt = len / 2;
    } else {
        lt = (len - 1) / 2 + 1;
    }

    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (thread_idz >= x.size(0) || thread_idy >= x.size(1) || thread_idx >= lt ) {
        return;
    }
    const int left_pos = thread_idx;
    y[thread_idz][thread_idy][left_pos] = 0;
    for (int i = 0; i <=left_pos; i++) {
        y[thread_idz][thread_idy][left_pos] += x[thread_idz][thread_idy][i] * weight[thread_idy][left_pos][i];
    }
    const int right_pos = len - thread_idx - 1;
    y[thread_idz][thread_idy][right_pos] = 0;
    for (int i = 0; i <=right_pos; i++) {
        y[thread_idz][thread_idy][right_pos] += x[thread_idz][thread_idy][i] * weight[thread_idy][right_pos][i];
    }
}


torch::Tensor weighted_cumsum_batch(torch::Tensor x, torch::Tensor w) {
    // Minimum required number of threads, assigns them dynamically to respective positions upon each iteration.
    // Results in uncoalesced writes, which is still faster than coalesced writes with half threads idling.

    TORCH_CHECK(x.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 3, "Input must be 3-dimensional as (B, H, S)");

    TORCH_CHECK(w.device().is_cuda(),   "weight must be a CUDA tensor");
    TORCH_CHECK(w.dim() == 3,           "weight must be 3-dimensional as (H, S, S)");
    TORCH_CHECK(w.size(2) == x.size(2), "weight dimensions must be compatible with the input");
    TORCH_CHECK(w.size(1) == w.size(2), "weight must be a square matrix");

    if (x.size(2) == 0) {
        return x;
    }

    //int maxThreadsDimZ = deviceProp.maxThreadsDim[2];


    auto y = x.clone();

    const int B = x.size(0);
    const int H = x.size(1);
    const int S = x.size(2);
    int St;
    if (S % 2 == 0) {
        St = S / 2;
    } else {
        St = (S - 1) / 2 + 1;
    }

    
    
    const int threads_S = getOptimalThreadsPerBlock(St, 0, 512); //only need half of the dimension. In each dimension, we fully fill O(S) computation
    const int threads_H = getOptimalThreadsPerBlock( H, 1, 1024/threads_S);
    const int threads_B = getOptimalThreadsPerBlock( B, 2, 1024/threads_S/threads_H);
    // thread has limit per each channel
    // and totally thread for one block is limited 1024
    const dim3 threads(threads_S, threads_H, threads_B);
    const dim3 blocks((St + threads_S - 1)/ threads_S, 
                      (H + threads_H - 1) / threads_H,
                      (B + threads_B - 1) / threads_B);
    // printf("Threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    // printf("Blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
    
    // cudaDeviceProp deviceProp;
    // int device;
    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&deviceProp, device);
    // printf("MaxThreads: (%d, %d, %d),(%d)\n", deviceProp.maxThreadsDim[0], 
    //                                      deviceProp.maxThreadsDim[1], 
    //                                      deviceProp.maxThreadsDim[2],
    //                                      deviceProp.maxThreadsPerBlock);



    // const dim3 threads(threads_S                     , B);
    // const dim3 blocks((St + threads_S - 1)/ threads_S, H);

    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "weighted_cumsum_kernel_batch", ([&] {
        weighted_cumsum_kernel_batch<scalar_t><<<blocks, threads>>>(
            y.packed_accessor32<scalar_t, 3>(),
            x.packed_accessor32<scalar_t, 3>(),
            w.packed_accessor32<scalar_t, 3>()
        );
    }));
    

    return y;
}

torch::Tensor weighted_cumsum_batch_cuda(torch::Tensor x, torch::Tensor w) {
    return weighted_cumsum_batch(x, w);
}
