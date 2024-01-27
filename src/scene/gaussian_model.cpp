#include "gaussian_model.h"
#include "src/utils/general_utils.h"
#include "src/arguments/params.h"

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

GaussianModel::GaussianModel(int sh_degree)
    : max_sh_degree_{sh_degree},
      percent_dense_{0},
      core_params_{
          0,               // active_sh_degree_,
          torch::empty(0), // xyz_
          torch::empty(0), // features_dc_
          torch::empty(0), // features_rest_
          torch::empty(0), // scaling_
          torch::empty(0), // rotation_
          torch::empty(0), // opacity_
          torch::empty(0), // max_radii2D_
          torch::empty(0), // xyz_gradient_accum_
          torch::empty(0), // denom_
          nullptr,         // optimizer_
          0,               // spatial_lr_scale_
      },
      scaling_activation_{&torch::exp},
      scaling_inverse_activation_{&torch::log},
      opacity_activation_{&torch::sigmoid},
      inverse_opacity_activation_{&torch::special::logit},
      rotation_activation_{&torch::nn::functional::normalize},
      covariance_activation_{build_covariance_from_scaling_rotation}
{
}

void GaussianModel::CoreParams::capture(const std::string &tensors_path, const std::string &opt_path)
{
    std::vector<torch::Tensor> tensors = {
        torch::tensor(active_sh_degree_),
        xyz_,
        features_dc_,
        features_rest_,
        scaling_,
        rotation_,
        opacity_,
        max_radii2D_,
        xyz_gradient_accum_,
        denom_,
        torch::tensor(spatial_lr_scale_),
    };
    torch::save(tensors, tensors_path);
    torch::save(*optimizer_, opt_path);
}

void GaussianModel::CoreParams::restore(const std::string &tensors_path, const std::string &opt_path)
{
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, tensors_path);
    active_sh_degree_ = tensors[0].item<int>();
    xyz_ = tensors[1];
    features_dc_ = tensors[2];
    features_rest_ = tensors[3];
    scaling_ = tensors[4];
    rotation_ = tensors[5];
    opacity_ = tensors[6];
    max_radii2D_ = tensors[7];
    xyz_gradient_accum_ = tensors[8];
    denom_ = tensors[9];
    spatial_lr_scale_ = tensors[10].item<float>();
    torch::load(*optimizer_, opt_path);
}

void GaussianModel::capture(const std::string &tensors_path, const std::string &opt_path)
{
    core_params_.capture(tensors_path, opt_path);
}

void GaussianModel::restore(const std::string &tensors_path, const std::string &opt_path)
{
    core_params_.restore(tensors_path, opt_path);
}

auto GaussianModel::get_scaling() -> torch::Tensor
{
    return scaling_activation_(core_params_.scaling_);
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
