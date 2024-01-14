#include <torch/torch.h>

class GaussianModel
{
private:
    int active_sh_degree_{0};
    int max_sh_degree_{};
    torch::Tensor xyz_{torch::empty(0)};
    torch::Tensor features_dc_{torch::empty(0)};
    torch::Tensor features_rest_{torch::empty(0)};
    torch::Tensor scaling_{torch::empty(0)};
    torch::Tensor rotation_{torch::empty(0)};
    torch::Tensor opacity{torch::empty(0)};
    torch::Tensor max_radii2D_{torch::empty(0)};
    torch::Tensor xyz_gradient_accum_{torch::empty(0)};
    torch::Tensor denom{torch::empty(0)};
    // optimizer = None
    float percent_dense_{0};
    float spatial_lr_scale_{0};

    typedef at::Tensor (*Activation_0)(const at::Tensor &);
    Activation_0 scaling_activation_;
    Activation_0 scaling_inverse_activation_;
    Activation_0 opacity_activation_;
    Activation_0 inverse_opacity_activation_;

    typedef at::Tensor (*Activation_1)(const at::Tensor &, torch::nn::functional::NormalizeFuncOptions);
    Activation_1 rotation_activation_;

    typedef torch::Tensor (*Activation_2)(
        const torch::Tensor &scaling,
        const int &scaling_modifier,
        const torch::Tensor &rotation);
    Activation_2 covariance_activation_;

public:
    GaussianModel(int sh_degree);

private:
    void setup_functions();
    auto get_scaling() -> torch::Tensor;
};