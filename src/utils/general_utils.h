#include <torch/torch.h>

auto build_scaling_rotation(const torch::Tensor &scaling, const torch::Tensor &rotation)
    -> torch::Tensor;

auto strip_symmetric(const torch::Tensor &sym) -> torch::Tensor;

auto get_expon_lr_func(
    float lr_init,
    float lr_final,
    int lr_delay_steps = 0,
    float lr_delay_mult = 1.0,
    int max_steps = 1000000) -> std::function<float(int)>;