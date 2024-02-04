#include "scene/gaussian_model.h"
#include "utils/general_utils.h"
#include "arguments/params.h"
#include <filesystem>
namespace fs = std::filesystem;

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

// test passed
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
    if (optimizer_ != nullptr)
    {
        torch::save(*optimizer_, opt_path);
    }
}

// test passed
void GaussianModel::CoreParams::restore(const std::string &tensors_path, const std::string &opt_path)
{
    if (fs::exists(tensors_path))
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
    }
    if (fs::exists(opt_path))
    {
        // l = [
        //{'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
        //{'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
        //{'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        //{'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        //{'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        //{'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        //]
        // self.optimizer = torch.optim.Adam(l, lr = 0.0, eps = 1e-15)
        torch::load(*optimizer_, opt_path);
    }
}

auto GaussianModel::get_core_params() -> GaussianModel::CoreParams &
{
    return core_params_;
}

auto GaussianModel::get_core_params() const -> const GaussianModel::CoreParams &
{
    return core_params_;
}

// test passed
void GaussianModel::capture(const std::string &tensors_path, const std::string &opt_path)
{
    core_params_.capture(tensors_path, opt_path);
}

// test passed
void GaussianModel::restore(const std::string &tensors_path, const std::string &opt_path)
{
    core_params_.restore(tensors_path, opt_path);
}

// test passed
auto GaussianModel::get_scaling() const -> torch::Tensor
{
    return scaling_activation_(core_params_.scaling_);
}

// test passed
auto GaussianModel::get_rotation() const -> torch::Tensor
{
    namespace F = torch::nn::functional;
    return rotation_activation_(core_params_.rotation_, F::NormalizeFuncOptions().dim(1).p(2));
}

// test passed
auto GaussianModel::get_xyz() const -> const torch::Tensor &
{
    return core_params_.xyz_;
}

// test passed
auto GaussianModel::get_features() const -> torch::Tensor
{
    return torch::cat({core_params_.features_dc_, core_params_.features_rest_}, 1);
}

// test passed
auto GaussianModel::get_opacity() const -> torch::Tensor
{
    return opacity_activation_(core_params_.opacity_);
}

// test passed
auto GaussianModel::get_covariance(int scaling_modifier) const -> torch::Tensor
{
    return covariance_activation_(get_scaling(), scaling_modifier, core_params_.rotation_);
}

// test passed
auto GaussianModel::oneup_SH_degree() -> void
{
    if (core_params_.active_sh_degree_ < max_sh_degree_)
    {
        core_params_.active_sh_degree_++;
    }
}

auto GaussianModel::setup(const OptimizationParams &params) -> void
{
    // percent_dense_ = params.percent_dense_;
    //      self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    //      self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    //    l = [
    //        {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
    //        {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
    //        {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
    //        {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
    //        {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
    //        {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
    //    ]

    //    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    //    self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
    //                                                lr_final=training_args.position_lr_final*self.spatial_lr_scale,
    //                                                lr_delay_mult=training_args.position_lr_delay_mult,
    //                                                max_steps=training_args.position_lr_max_steps)
}

auto GaussianModel::update_learning_rate(int iteration) const -> float
{
    for (const auto &param_group : core_params_.optimizer_->param_groups())
    {
        // if (param_group["name"] == "xyz")
        //{
        //     auto lr = 0.0; // self.xyz_scheduler_args(iteration)
        //     param_group["lr"] = lr;
        //     return lr;
        // }
    }
    // for param_group in self.optimizer.param_groups:
    //         if param_group["name"] == "xyz":
    //             lr = self.xyz_scheduler_args(iteration)
    //             param_group['lr'] = lr
    //             return lr
    return 0.0;
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

    void test_capture_and_restore()
    {
        std::cout << " test_capture_and_restore" << std::endl;
        GaussianModel model(2);
        model.capture("test_gaussian_model.pt", "test_gaussian_model_opt.pt");
        model.restore("test_gaussian_model.pt", "test_gaussian_model_opt.pt");
        const auto &core_params = model.get_core_params();
        BOOST_CHECK_EQUAL(core_params.active_sh_degree_, 0);
        auto answer = torch::empty(0);
        BOOST_CHECK(torch::allclose(answer, core_params.xyz_));
        BOOST_CHECK(torch::allclose(answer, core_params.features_dc_));
        BOOST_CHECK(torch::allclose(answer, core_params.scaling_));
        BOOST_CHECK(torch::allclose(answer, core_params.rotation_));
        BOOST_CHECK(torch::allclose(answer, core_params.opacity_));
        BOOST_CHECK(torch::allclose(answer, core_params.max_radii2D_));
        BOOST_CHECK(torch::allclose(answer, core_params.xyz_gradient_accum_));
        BOOST_CHECK(torch::allclose(answer, core_params.denom_));
        BOOST_CHECK(core_params.optimizer_ == nullptr);
        BOOST_CHECK_EQUAL(core_params.spatial_lr_scale_, 0);
    }

    void test_get_scaling()
    {
        std::cout << " test_get_scaling" << std::endl;
        GaussianModel model(2);
        auto &core_params = model.get_core_params();
        core_params.scaling_ = torch::zeros({2, 3});
        auto scaling = model.get_scaling();
        auto answer = torch::ones({2, 3});
        BOOST_CHECK(torch::allclose(answer, scaling));
    }

    void test_get_rotation()
    {
        std::cout << " test_get_rotation" << std::endl;
        GaussianModel model(2);
        auto &core_params = model.get_core_params();
        core_params.rotation_ = torch::ones({2, 4});
        auto rotation = model.get_rotation();
        auto answer = 0.5 * torch::ones({2, 4});
        BOOST_CHECK(torch::allclose(answer, rotation));
    }

    void test_get_xyz()
    {
        std::cout << " test_get_xyz" << std::endl;
        GaussianModel model(2);
        auto &core_params = model.get_core_params();
        core_params.xyz_ = torch::ones({2, 4});
        auto xyz = model.get_xyz();
        auto answer = torch::ones({2, 4});
        BOOST_CHECK(torch::allclose(answer, xyz));
    }

    void test_get_features()
    {
        std::cout << " test_get_features" << std::endl;
        GaussianModel model(2);
        auto &core_params = model.get_core_params();
        core_params.features_dc_ = torch::ones({2, 4});
        core_params.features_rest_ = torch::zeros({2, 4});
        auto features = model.get_features();
        auto answer = torch::cat({torch::ones({2, 4}), torch::zeros({2, 4})}, 1);
        BOOST_CHECK(torch::allclose(answer, features));
    }

    void test_get_opacity()
    {
        std::cout << " test_get_opacity" << std::endl;
        GaussianModel model(2);
        auto &core_params = model.get_core_params();
        core_params.opacity_ = torch::zeros({2, 4});
        auto opacity = model.get_opacity();
        auto answer = 0.5 * torch::ones({2, 4});
        BOOST_CHECK(torch::allclose(answer, opacity));
    }

    void test_get_covariance()
    {
        std::cout << " test_get_covarinance" << std::endl;
        GaussianModel model(2);
        auto &core_params = model.get_core_params();
        core_params.scaling_ = torch::zeros({2, 3}); // 1.0
        core_params.rotation_ = torch::ones({2, 4});
        auto covariance = model.get_covariance().to(torch::kCPU);
        auto answer = torch::tensor({{1.0, 0.0, 0.0, 1.0, 0.0, 1.0}, {1.0, 0.0, 0.0, 1.0, 0.0, 1.0}});
        BOOST_CHECK(torch::allclose(answer, covariance));
    }

    void test_oneup_SH_degree()
    {
        std::cout << " test_oneup_SH_degree" << std::endl;
        GaussianModel model(2);
        auto &core_params = model.get_core_params();
        core_params.active_sh_degree_ = 0;
        model.oneup_SH_degree();
        BOOST_CHECK_EQUAL(core_params.active_sh_degree_, 1);
    }

}

BOOST_AUTO_TEST_CASE(test_gaussian_model)
{
    std::cout << "test_gaussian_model" << std::endl;
    test_build_covariance_from_scaling_rotation();
    test_capture_and_restore();
    test_get_scaling();
    test_get_rotation();
    test_get_xyz();
    test_get_features();
    test_get_opacity();
    test_get_covariance();
    test_oneup_SH_degree();
}
#endif // UNIT_TEST
