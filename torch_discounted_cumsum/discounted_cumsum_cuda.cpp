#include <torch/extension.h>


torch::Tensor discounted_cumsum_left_cuda(torch::Tensor x, torch::Tensor gamma);
torch::Tensor discounted_cumsum_right_cuda(torch::Tensor x, torch::Tensor gamma);
torch::Tensor discounted_cumsum3_left_cuda(torch::Tensor x, torch::Tensor gamma);
torch::Tensor discounted_cumsum3_right_cuda(torch::Tensor x, torch::Tensor gamma);
torch::Tensor weighted_cumsum_cuda(torch::Tensor x, torch::Tensor w);
torch::Tensor weighted_cumsum_batch_cuda(torch::Tensor x, torch::Tensor w);
torch::Tensor qkvg_retention_cuda(torch::Tensor q, 
                                  torch::Tensor k,
                                  torch::Tensor v,
                                  torch::Tensor g);




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("discounted_cumsum_left_cuda", &discounted_cumsum_left_cuda, "Discounted Cumulative Sums CUDA (Left)");
    m.def("discounted_cumsum_right_cuda", &discounted_cumsum_right_cuda, "Discounted Cumulative Sums CUDA (Right)");
    m.def("discounted_cumsum3_left_cuda", &discounted_cumsum3_left_cuda, "Discounted Cumulative 3 Sums CUDA (Left)");
    m.def("discounted_cumsum3_right_cuda", &discounted_cumsum3_right_cuda, "Discounted Cumulative 3 Sums CUDA (Right)");
    m.def("weighted_cumsum_cuda", &weighted_cumsum_cuda, "Weighted Cumulative Sums CUDA (Right)");
    m.def("weighted_cumsum_batch_cuda", &weighted_cumsum_batch_cuda, "Weighted Cumulative Sums CUDA (Right)");
    m.def("qkvg_retention_cuda", &qkvg_retention_cuda, "qkvg_retention_cuda");
}

