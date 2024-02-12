#include <torch/torch.h>
#include <memory>
#include <functional>
struct OptimizationParams;

class GaussianModel
{
public:
    struct CoreParams
    {
        int active_sh_degree_;
        torch::Tensor xyz_;
        torch::Tensor features_dc_;
        torch::Tensor features_rest_;
        torch::Tensor scaling_;
        torch::Tensor rotation_;
        torch::Tensor opacity_;
        torch::Tensor max_radii2D_;
        torch::Tensor xyz_gradient_accum_;
        torch::Tensor denom_;
        std::map<std::string, std::unique_ptr<torch::optim::Adam>> optimizers_;
        float spatial_lr_scale_;

        void capture(
            const std::string &path,
            const std::string &opt_path_for_xyz,
            const std::string &opt_path_for_f_dc,
            const std::string &opt_path_for_f_rest,
            const std::string &opt_path_for_opacity,
            const std::string &opt_path_for_scaling,
            const std::string &opt_path_for_rotation);
        void restore(
            const std::string &path,
            const std::string &opt_path_for_xyz,
            const std::string &opt_path_for_f_dc,
            const std::string &opt_path_for_f_rest,
            const std::string &opt_path_for_opacity,
            const std::string &opt_path_for_scaling,
            const std::string &opt_path_for_rotation);
    };

private:
    int max_sh_degree_;
    float percent_dense_;

    CoreParams core_params_;

    typedef std::function<torch::Tensor(const torch::Tensor &)> Activation_0;
    Activation_0 scaling_activation_;
    Activation_0 scaling_inverse_activation_;
    Activation_0 opacity_activation_;
    Activation_0 inverse_opacity_activation_;

    typedef std::function<at::Tensor(const torch::Tensor &, torch::nn::functional::NormalizeFuncOptions)> Activation_1;
    Activation_1 rotation_activation_;

    typedef std::function<torch::Tensor(const torch::Tensor &, const int &, const torch::Tensor &)> Activation_2;
    Activation_2 covariance_activation_;

    std::function<float(int)> xyz_scheduler_args_;

public:
    GaussianModel(int sh_degree);

    void capture(
        const std::string &tensors_path,
        const std::string &opt_path_for_xyz,
        const std::string &opt_path_for_f_dc,
        const std::string &opt_path_for_f_rest,
        const std::string &opt_path_for_opacity,
        const std::string &opt_path_for_scaling,
        const std::string &opt_path_for_rotation);
    void restore(
        const OptimizationParams &params,
        const std::string &tensors_path,
        const std::string &opt_path_for_xyz,
        const std::string &opt_path_for_f_dc,
        const std::string &opt_path_for_f_rest,
        const std::string &opt_path_for_opacity,
        const std::string &opt_path_for_scaling,
        const std::string &opt_path_for_rotation);
    auto get_core_params() -> CoreParams &;
    auto get_core_params() const -> const CoreParams &;

    auto get_scaling() const -> torch::Tensor;
    auto get_rotation() const -> torch::Tensor;
    auto get_xyz() const -> const torch::Tensor &;
    auto get_features() const -> torch::Tensor;
    auto get_opacity() const -> torch::Tensor;
    auto get_covariance(int scaling_modifier = 1) const -> torch::Tensor;

    auto oneup_SH_degree() -> void;
    auto setup(const OptimizationParams &params) -> void;
    auto update_learning_rate(int iteration) -> float;

    auto get_max_sh_degree() const -> int;

    Activation_0 get_scaling_activation() const;
    Activation_0 get_scaling_inverse_activation() const;
    Activation_0 get_opacity_activation() const;
    Activation_0 get_inverse_opacity_activation() const;
    Activation_1 get_rotation_activation() const;
    Activation_2 get_covariance_activation() const;
    std::function<float(int)> get_xyz_scheduler_args() const;
};