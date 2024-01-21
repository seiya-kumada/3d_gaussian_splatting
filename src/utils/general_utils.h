#include <torch/torch.h>

auto build_scaling_rotation(const torch::Tensor &scaling, const torch::Tensor &rotation)
    -> torch::Tensor;

auto strip_symmetric(const torch::Tensor &sym) -> torch::Tensor;