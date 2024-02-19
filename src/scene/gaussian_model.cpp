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

// test passed
/**
 * @brief Constructs a GaussianModel object with the specified spherical harmonic degree.
 *
 * @param sh_degree The spherical harmonic degree of the GaussianModel.
 */
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
          {},              // optimizers_
          0,               // spatial_lr_scale_
      },
      scaling_activation_{&torch::exp},
      scaling_inverse_activation_{&torch::log},
      opacity_activation_{&torch::sigmoid},
      inverse_opacity_activation_{&torch::special::logit},
      rotation_activation_{&torch::nn::functional::normalize},
      covariance_activation_{build_covariance_from_scaling_rotation},
      xyz_scheduler_args_{get_expon_lr_func(0.0, 0.0, 0, 1.0, 1000000)}
{
}

// test passed
/**
 * @brief Captures the core parameters of the Gaussian model.
 *
 * @param tensors_path The path to the tensors.
 * @param opt_path_for_xyz The path for XYZ optimizer.
 * @param opt_path_for_f_dc The path for F_DC optimizer.
 * @param opt_path_for_f_rest The path for F_REST optimizer.
 * @param opt_path_for_opacity The path for opacity optimizer.
 * @param opt_path_for_scaling The path for scaling optimizer.
 * @param opt_path_for_rotation The path for rotation optimizer.
 */
void GaussianModel::CoreParams::capture(
    const std::string &tensors_path,
    const std::string &opt_path_for_xyz,
    const std::string &opt_path_for_f_dc,
    const std::string &opt_path_for_f_rest,
    const std::string &opt_path_for_opacity,
    const std::string &opt_path_for_scaling,
    const std::string &opt_path_for_rotation)
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

    for (auto &[name, optimizer] : optimizers_)
    {
        if (optimizer != nullptr)
        {
            if (name == "xyz")
            {
                torch::save(*optimizer, opt_path_for_xyz);
            }
            else if (name == "f_dc")
            {
                ;
                torch::save(*optimizer, opt_path_for_f_dc);
            }
            else if (name == "f_rest")
            {
                torch::save(*optimizer, opt_path_for_f_rest);
            }
            else if (name == "opacity")
            {
                torch::save(*optimizer, opt_path_for_opacity);
            }
            else if (name == "scaling")
            {
                torch::save(*optimizer, opt_path_for_scaling);
            }
            else if (name == "rotation")
            {
                torch::save(*optimizer, opt_path_for_rotation);
            }
        }
    }
}

// test passed
auto GaussianModel::get_max_sh_degree() const -> int
{
    return max_sh_degree_;
}
// test passed
/**
 * @brief Restores the core parameters of the GaussianModel.
 *
 * This function restores the core parameters of the GaussianModel from the specified files.
 *
 * @param tensors_path The path to the tensors.
 * @param opt_path_for_xyz The path for XYZ optimizer.
 * @param opt_path_for_f_dc The path for F_DC optimizer.
 * @param opt_path_for_f_rest The path for F_REST optimizer.
 * @param opt_path_for_opacity The path for opacity optimizer.
 * @param opt_path_for_scaling The path for scaling optimizer.
 * @param opt_path_for_rotation The path for rotation optimizer.
 */
void GaussianModel::CoreParams::restore(
    const std::string &tensors_path,
    const std::string &opt_path_for_xyz,
    const std::string &opt_path_for_f_dc,
    const std::string &opt_path_for_f_rest,
    const std::string &opt_path_for_opacity,
    const std::string &opt_path_for_scaling,
    const std::string &opt_path_for_rotation)
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

    if (fs::exists(opt_path_for_xyz))
    {
        torch::load(*optimizers_["xyz"], opt_path_for_xyz);
    }
    if (fs::exists(opt_path_for_f_dc))
    {
        torch::load(*optimizers_["f_dc"], opt_path_for_f_dc);
    }
    if (fs::exists(opt_path_for_f_rest))
    {
        torch::load(*optimizers_["f_rest"], opt_path_for_f_rest);
    }
    if (fs::exists(opt_path_for_opacity))
    {
        torch::load(*optimizers_["opacity"], opt_path_for_opacity);
    }
    if (fs::exists(opt_path_for_scaling))
    {
        torch::load(*optimizers_["scaling"], opt_path_for_scaling);
    }
    if (fs::exists(opt_path_for_rotation))
    {
        torch::load(*optimizers_["rotation"], opt_path_for_rotation);
    }
}

// test passed
auto GaussianModel::get_core_params() -> GaussianModel::CoreParams &
{
    return core_params_;
}

// test passed
auto GaussianModel::get_core_params() const -> const GaussianModel::CoreParams &
{
    return core_params_;
}

// test passed
/**
 * Captures the Gaussian model by saving the tensors and optimizer.
 *
 * @param tensors_path The path to save the tensors.
 * @param opt_path_for_xyz The path to save the optimizer for XYZ.
 * @param opt_path_for_f_dc The path to save the optimizer for F_DC.
 * @param opt_path_for_f_rest The path to save the optimizer for F_REST.
 * @param opt_path_for_opacity The path to save the optimizer for opacity.
 * @param opt_path_for_scaling The path to save the optimizer for scaling.
 * @param opt_path_for_rotation The path to save the optimizer for rotation.
 */
void GaussianModel::capture(
    const std::string &tensors_path,
    const std::string &opt_path_for_xyz,
    const std::string &opt_path_for_f_dc,
    const std::string &opt_path_for_f_rest,
    const std::string &opt_path_for_opacity,
    const std::string &opt_path_for_scaling,
    const std::string &opt_path_for_rotation)
{
    core_params_.capture(
        tensors_path,
        opt_path_for_xyz,
        opt_path_for_f_dc,
        opt_path_for_f_rest,
        opt_path_for_opacity,
        opt_path_for_scaling,
        opt_path_for_rotation);
}

// test passed
void GaussianModel::restore(
    const OptimizationParams &params,
    const std::string &tensors_path,
    const std::string &opt_path_for_xyz,
    const std::string &opt_path_for_f_dc,
    const std::string &opt_path_for_f_rest,
    const std::string &opt_path_for_opacity,
    const std::string &opt_path_for_scaling,
    const std::string &opt_path_for_rotation)
{
    setup(params);
    core_params_.restore(
        tensors_path,
        opt_path_for_xyz,
        opt_path_for_f_dc,
        opt_path_for_f_rest,
        opt_path_for_opacity,
        opt_path_for_scaling,
        opt_path_for_rotation);
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

//  test passed
auto GaussianModel::setup(const OptimizationParams &training_args) -> void
{
    percent_dense_ = training_args.percent_dense_;
    core_params_.xyz_gradient_accum_ = torch::zeros((get_xyz().size(0), 1), torch::device(torch::kCUDA));
    core_params_.denom_ = torch::zeros({get_xyz().size(0), 1}, torch::device(torch::kCUDA));

    // オプティマイザの定義、パラメータグループを追加
    core_params_.optimizers_["xyz"] = std::make_unique<torch::optim::Adam>(
        std::vector<torch::Tensor>{core_params_.xyz_},
        torch::optim::AdamOptions{training_args.position_lr_init_ * core_params_.spatial_lr_scale_});

    core_params_.optimizers_["f_dc"] = std::make_unique<torch::optim::Adam>(
        std::vector<torch::Tensor>{core_params_.features_dc_},
        torch::optim::AdamOptions{training_args.feature_lr_});

    core_params_.optimizers_["f_rest"] = std::make_unique<torch::optim::Adam>(
        std::vector<torch::Tensor>{core_params_.features_rest_},
        torch::optim::AdamOptions{training_args.feature_lr_ / 20.0});

    core_params_.optimizers_["opacity"] = std::make_unique<torch::optim::Adam>(
        std::vector<torch::Tensor>{core_params_.opacity_},
        torch::optim::AdamOptions{training_args.opacity_lr_});

    core_params_.optimizers_["scaling"] = std::make_unique<torch::optim::Adam>(
        std::vector<torch::Tensor>{core_params_.scaling_},
        torch::optim::AdamOptions{training_args.scaling_lr_});

    core_params_.optimizers_["rotation"] = std::make_unique<torch::optim::Adam>(
        std::vector<torch::Tensor>{core_params_.rotation_},
        torch::optim::AdamOptions{training_args.rotation_lr_});

    xyz_scheduler_args_ = get_expon_lr_func(
        training_args.position_lr_init_ * core_params_.spatial_lr_scale_,
        training_args.position_lr_final_ * core_params_.spatial_lr_scale_,
        training_args.position_lr_delay_mult_,
        training_args.position_lr_max_steps_);
}

// test passed
auto GaussianModel::update_learning_rate(int iteration) -> float
{
    auto &param_group = core_params_.optimizers_["xyz"]->param_groups()[0];
    auto lr = xyz_scheduler_args_(iteration);
    static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(lr);
    return lr;
}

// test passed
GaussianModel::Activation_0 GaussianModel::get_scaling_activation() const
{
    return scaling_activation_;
}

// test passed
GaussianModel::Activation_0 GaussianModel::get_scaling_inverse_activation() const
{
    return scaling_inverse_activation_;
}

// test passed
GaussianModel::Activation_0 GaussianModel::get_opacity_activation() const
{
    return opacity_activation_;
}

// test passed
GaussianModel::Activation_0 GaussianModel::get_inverse_opacity_activation() const
{
    return inverse_opacity_activation_;
}

// test passed
GaussianModel::Activation_1 GaussianModel::get_rotation_activation() const
{
    return rotation_activation_;
}

// test passed
GaussianModel::Activation_2 GaussianModel::get_covariance_activation() const
{
    return covariance_activation_;
}

// test passed
auto GaussianModel::get_xyz_scheduler_args() const -> std::function<float(int)>
{
    return xyz_scheduler_args_;
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
        auto params = OptimizationParams{};

        model.capture(
            "test_gaussian_model.pt",
            "test_gaussian_model_opt_for_xzy.pt",
            "test_gaussian_model_opt_for_f_dc.pt",
            "test_gaussian_model_opt_for_f_rest.pt",
            "test_gaussian_model_opt_for_opacity.pt",
            "test_gaussian_model_opt_for_scaling.pt",
            "test_gaussian_model_opt_for_rotation.pt");
        model.restore(
            params,
            "test_gaussian_model.pt",
            "test_gaussian_model_opt_for_xzy.pt",
            "test_gaussian_model_opt_for_f_dc.pt",
            "test_gaussian_model_opt_for_f_rest.pt",
            "test_gaussian_model_opt_for_opacity.pt",
            "test_gaussian_model_opt_for_scaling.pt",
            "test_gaussian_model_opt_for_rotation.pt");
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
        for (auto &[name, optimizer] : core_params.optimizers_)
        {
            BOOST_CHECK(optimizer != nullptr);
        }
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

    void test_setup()
    {
        std::cout << " test_setup" << std::endl;
        auto params = OptimizationParams{};
        GaussianModel model(2);
        model.setup(params);
        auto &core_params = model.get_core_params();
        for (auto &[name, optimizer] : core_params.optimizers_)
        {
            BOOST_CHECK(optimizer != nullptr);
        }
    }

    void test_constructor()
    {
        std::cout << " test_constructor" << std::endl;
        GaussianModel model(2);
        BOOST_CHECK_EQUAL(model.get_max_sh_degree(), 2);
        const auto &core_params = model.get_core_params();
        BOOST_CHECK_EQUAL(core_params.active_sh_degree_, 0);
        auto answer = torch::empty(0);
        BOOST_CHECK(torch::allclose(answer, core_params.xyz_));
        BOOST_CHECK(torch::allclose(answer, core_params.features_dc_));
        BOOST_CHECK(torch::allclose(answer, core_params.features_rest_));
        BOOST_CHECK(torch::allclose(answer, core_params.scaling_));
        BOOST_CHECK(torch::allclose(answer, core_params.rotation_));
        BOOST_CHECK(torch::allclose(answer, core_params.opacity_));
        BOOST_CHECK(torch::allclose(answer, core_params.max_radii2D_));
        BOOST_CHECK(torch::allclose(answer, core_params.xyz_gradient_accum_));
        BOOST_CHECK(torch::allclose(answer, core_params.denom_));
        BOOST_CHECK_EQUAL(core_params.optimizers_.size(), 0);
        BOOST_CHECK_EQUAL(core_params.spatial_lr_scale_, 0);

        auto &core_params_ = model.get_core_params();
        core_params_.spatial_lr_scale_ = 1;
        BOOST_CHECK_EQUAL(core_params_.spatial_lr_scale_, 1);

        {
            auto scaling_activation = model.get_scaling_activation();
            auto v = torch::ones({1});
            BOOST_CHECK_CLOSE(scaling_activation(v).item<float>(), 2.718281828459045, 1e-5);
        }
        {
            auto scaling_inverse_activation = model.get_scaling_inverse_activation();
            auto v = torch::ones({1});
            BOOST_CHECK_CLOSE(scaling_inverse_activation(v).item<float>(), 0, 1e-5);
        }

        {
            auto opacity_activation = model.get_opacity_activation();
            auto u = torch::zeros({1});
            BOOST_CHECK_CLOSE(opacity_activation(u).item<float>(), 0.5, 1e-5);
        }
        {
            auto inverse_opacity_activation = model.get_inverse_opacity_activation();
            auto s = 0.5 * torch::ones({1});
            BOOST_CHECK_CLOSE(inverse_opacity_activation(s).item<float>(), 0, 1e-5);
        }

        {
            auto rotation_activation = model.get_rotation_activation();
            auto t = torch::ones({1, 4});
            namespace F = torch::nn::functional;
            BOOST_CHECK_CLOSE(rotation_activation(t, F::NormalizeFuncOptions().dim(1).p(2)).index({0, 0}).item<float>(), 0.5, 1e-5);
        }
        {
            auto covariance_activation = model.get_covariance_activation();
            auto device = torch::kCUDA;
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
            auto cov = covariance_activation(s, scaling_modifier, r);
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

        {
            auto xyz_scheduler_args = model.get_xyz_scheduler_args();
            BOOST_CHECK_CLOSE(xyz_scheduler_args(0), 0, 1e-5);
        }
    }

    void test_update_learning_rate()
    {
        std::cout << " test_update_learning_rate" << std::endl;
        auto params = OptimizationParams{};

        GaussianModel model(2);
        auto &core_params = model.get_core_params();
        core_params.spatial_lr_scale_ = 1;
        model.setup(params);

        const auto &param_group_before = model.get_core_params().optimizers_["xyz"]->param_groups()[0];
        auto lr_before = static_cast<const torch::optim::AdamOptions &>(param_group_before.options()).lr();
        BOOST_CHECK_CLOSE(lr_before, 0.000159999, 1e-3);

        auto lr_after = model.update_learning_rate(-1);
        BOOST_CHECK_CLOSE(lr_after, 0.0, 1e-3);

        auto lr_after_ = static_cast<const torch::optim::AdamOptions &>(param_group_before.options()).lr();
        BOOST_CHECK_CLOSE(lr_after_, 0.0, 1e-3);
    }
}

BOOST_AUTO_TEST_CASE(test_gaussian_model)
{
    std::cout << "test_gaussian_model" << std::endl;
    test_build_covariance_from_scaling_rotation();
    test_get_scaling();
    test_get_rotation();
    test_get_xyz();
    test_get_features();
    test_get_opacity();
    test_get_covariance();
    test_oneup_SH_degree();
    test_setup();
    test_capture_and_restore();
    test_constructor();
    test_update_learning_rate();
}
#endif // UNIT_TEST
