#include "gaussian_model.h"
#include "src/utils/general_utils.h"

GaussianModel::GaussianModel(int sh_degree)
    : max_sh_degree_{sh_degree}
{
    setup_functions();
}

namespace
{

    torch::Tensor strip_lowerdiag(const torch::Tensor &L)
    {
        auto device = torch::kCUDA;
        torch::Tensor uncertainty = torch::zeros({L.size(0), 6}, torch::TensorOptions().dtype(torch::kFloat).device(device));

        uncertainty.select(1, 0) = L.select(1, 0).select(1, 0);
        uncertainty.select(1, 1) = L.select(1, 0).select(1, 1);
        uncertainty.select(1, 2) = L.select(1, 0).select(1, 2);
        uncertainty.select(1, 3) = L.select(1, 1).select(1, 1);
        uncertainty.select(1, 4) = L.select(1, 1).select(1, 2);
        uncertainty.select(1, 5) = L.select(1, 2).select(1, 2);

        return uncertainty;
    }

    torch::Tensor strip_symmetric(const torch::Tensor &sym)
    {
        return strip_lowerdiag(sym);
    }

    /**
     * @brief Builds a covariance matrix from scaling, scaling modifier, and rotation tensors.
     *
     * @param scaling The scaling tensor.
     * @param scaling_modifier The scaling modifier tensor.
     * @param rotation The rotation tensor.
     * @return The covariance matrix tensor.
     */
    torch::Tensor build_covariance_from_scaling_rotation(
        const torch::Tensor &scaling,
        const int &scaling_modifier,
        const torch::Tensor &rotation)
    {
        auto L = build_scaling_rotation(scaling_modifier * scaling, rotation);
        auto actual_covariance = torch::matmul(L, L.transpose(1, 2));
        auto symm = strip_symmetric(actual_covariance);
        return symm;
    }
}

void GaussianModel::setup_functions()
{
    scaling_activation_ = &torch::exp;
    scaling_inverse_activation_ = &torch::log;
    covariance_activation_ = build_covariance_from_scaling_rotation;
    opacity_activation_ = &torch::sigmoid;
    inverse_opacity_activation_ = &torch::special::logit;
    rotation_activation_ = &torch::nn::functional::normalize;
}
// self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

auto GaussianModel::get_scaling() -> torch::Tensor
{
    return scaling_activation_(scaling_);
}

#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(test_gaussian_model)
{
    BOOST_CHECK_EQUAL(1, 1);
}
#endif // UNIT_TEST
