#include "gaussian_model.h"
#include "src/utils/general_utils.h"

GaussianModel::GaussianModel(int sh_degree)
    : max_sh_degree_{sh_degree}
{
    setup_functions();
}

namespace
{
    // test passed
    /**
     * @brief Builds a covariance matrix from scaling, scaling modifier, and rotation tensors.
     *
     * @param scaling The scaling tensor whose shape is (N, 3).
     * @param scaling_modifier The scaling modifier tensor.
     * @param rotation The rotation tensor whose shape is (N, 4).
     * @return The covariance matrix tensor whose shape is (N, 6).
     */
    torch::Tensor build_covariance_from_scaling_rotation(
        const torch::Tensor &scaling,
        const int &scaling_modifier,
        const torch::Tensor &rotation)
    {
        auto L = build_scaling_rotation(scaling_modifier * scaling, rotation);
        auto actual_covariance = torch::matmul(L, L.transpose(1, 2));
        // actual_covariance.sizes(): (N, 3, 3)
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
namespace
{
    void test_build_covariance_from_scaling_rotation()
    {
        std::cout << " test_build_covariance_from_scaling_rotation" << std::endl;
        auto device = torch::kCUDA;
        // s = (0.5, 0.5, 0.5)
        torch::Tensor s = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device));
        s.index_put_({0, 0}, 0.5);
        s.index_put_({0, 1}, 0.5);
        s.index_put_({0, 2}, 0.5);
        s.index_put_({1, 0}, 0.25);
        s.index_put_({1, 1}, 0.25);
        s.index_put_({1, 2}, 0.25);
        // scaling_modifier = 1
        int scaling_modifier = 1;
        // r = (0.5, 0.5, 0.5, 0.5)
        torch::Tensor r = torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kFloat).device(device));
        r.index_put_({0, 0}, 0.5);
        r.index_put_({0, 1}, 0.5);
        r.index_put_({0, 2}, 0.5);
        r.index_put_({0, 3}, 0.5);
        r.index_put_({1, 0}, 0.25);
        r.index_put_({1, 1}, 0.25);
        r.index_put_({1, 2}, 0.25);
        r.index_put_({1, 3}, 0.25);
        auto cov = build_covariance_from_scaling_rotation(s, scaling_modifier, r);
        // std::cout << cov.sizes() << std::endl;
        BOOST_CHECK_EQUAL(cov.size(0), 2);
        BOOST_CHECK_EQUAL(cov.size(1), 6);
        BOOST_CHECK_EQUAL(cov.dtype(), torch::kFloat);
        BOOST_CHECK_EQUAL(cov.device().type(), torch::kCUDA);

        BOOST_CHECK_EQUAL(cov.index({0, 0}).item<float>(), 0.25);
        BOOST_CHECK_EQUAL(cov.index({0, 1}).item<float>(), 0);
        BOOST_CHECK_EQUAL(cov.index({0, 2}).item<float>(), 0);
        BOOST_CHECK_EQUAL(cov.index({0, 3}).item<float>(), 0.25);
        BOOST_CHECK_EQUAL(cov.index({0, 4}).item<float>(), 0);
        BOOST_CHECK_EQUAL(cov.index({0, 5}).item<float>(), 0.25);

        BOOST_CHECK_EQUAL(cov.index({1, 0}).item<float>(), 0.0625);
        BOOST_CHECK_EQUAL(cov.index({1, 1}).item<float>(), 0);
        BOOST_CHECK_EQUAL(cov.index({1, 2}).item<float>(), 0);
        BOOST_CHECK_EQUAL(cov.index({1, 3}).item<float>(), 0.0625);
        BOOST_CHECK_EQUAL(cov.index({1, 4}).item<float>(), 0);
        BOOST_CHECK_EQUAL(cov.index({1, 5}).item<float>(), 0.0625);
    }
}

BOOST_AUTO_TEST_CASE(test_gaussian_model)
{
    std::cout << "test_gaussian_model" << std::endl;
    test_build_covariance_from_scaling_rotation();
}
#endif // UNIT_TEST
