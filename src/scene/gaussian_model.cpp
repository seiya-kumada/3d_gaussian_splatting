#include "gaussian_model.h"

GaussianModel::GaussianModel(int sh_degree)
    : max_sh_degree_{sh_degree}
{
    setup_functions();
}

namespace
{
    void build_covariance_from_scaling_rotation(
        const torch::Tensor &scaling,
        int scaling_modifier,
        const torch::Tensor &rotation)
    {
    }
}

void GaussianModel::setup_functions()
{
    scaling_activation_ = &torch::exp;
    scaling_inverse_activation_ = &torch::log;
    // covariance_activation_ = &torch::exp;
    opacity_activation_ = &torch::sigmoid;
    inverse_opacity_activation_ = &torch::special::logit;
    rotation_activation_ = &torch::nn::functional::normalize;
}
// self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

auto GaussianModel::get_scaling() -> torch::Tensor
{
    return scaling_activation_(scaling_);
}