#include <torch/torch.h>
#include <memory>

class GaussianModel
{
public:
    struct CoreParams
    {
        int active_sh_degree_;                          //
        torch::Tensor xyz_;                             //
        torch::Tensor features_dc_;                     //
        torch::Tensor features_rest_;                   //
        torch::Tensor scaling_;                         //
        torch::Tensor rotation_;                        //
        torch::Tensor opacity_;                         //
        torch::Tensor max_radii2D_;                     //
        torch::Tensor xyz_gradient_accum_;              //
        torch::Tensor denom_;                           //
        std::unique_ptr<torch::optim::Adam> optimizer_; //
        float spatial_lr_scale_;                        //

        void capture(const std::string &path, const std::string &opt_path);
        void restore(const std::string &tensors_path, const std::string &opt_path);
    };

private:
    int max_sh_degree_;
    float percent_dense_;

    CoreParams core_params_;

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
    void capture(const std::string &tensors_path, const std::string &opt_path);
    void restore(const std::string &tensors_path, const std::string &opt_path);

private:
    auto get_scaling() -> torch::Tensor;
};