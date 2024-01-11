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

public:
    GaussianModel(int sh_degree);

private:
    void setup_functions();
};