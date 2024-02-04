#include <string>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>
#include <iostream>

template <typename T>
inline void print_param(const std::string &param_name, const T &param_value, std::ostream &out)
{
    out << std::boolalpha;
    out << " " << param_name << ": " << param_value << std::endl;
}

template <>
inline void print_param<std::vector<int>>(const std::string &param_name, const std::vector<int> &param_value, std::ostream &out)
{
    out << std::boolalpha;
    out << " " << param_name << ": ";
    for (auto &v : param_value)
    {
        out << v << " ";
    }
    out << std::endl;
}

struct ModelParams
{
    int sh_degree_{};
    std::string source_path_{};
    std::string model_path_{};
    std::string images_{"images"};
    int resolution_{-1};
    bool white_background_{false};
    std::string data_device_{"cuda"};
    bool eval_{false};

    void print_params(std::ostream &out = std::cout) const
    {
        out << "> Model parameters:" << std::endl;
        print_param(" sh_degree", sh_degree_, out);
        print_param(" source_path", source_path_, out);
        print_param(" model_path", model_path_, out);
        print_param(" images", images_, out);
        print_param(" resolution", resolution_, out);
        print_param(" white_background", white_background_, out);
        print_param(" data_device", data_device_, out);
        print_param(" eval", eval_, out);
    }
};

struct OptimizationParams
{
    int iterations_{30'000};
    float position_lr_init_{0.00016};
    float position_lr_final_{0.0000016};
    float position_lr_delay_mult_{0.01};
    int position_lr_max_steps_{30'000};
    float feature_lr_{0.0025};
    float opacity_lr_{0.05};
    float scaling_lr_{0.005};
    float rotation_lr_{0.001};
    float percent_dense_{0.01};
    float lambda_dssim_{0.2};
    int densification_interval_{100};
    int opacity_reset_interval_{3000};
    int densify_from_iter_{500};
    int densify_until_iter_{15'000};
    float densify_grad_threshold_{0.0002};
    bool random_background_{false};

    void print_params(std::ostream &out = std::cout) const
    {
        out << "> Optimization parameters:" << std::endl;
        print_param(" iterations", iterations_, out);
        print_param(" position_lr_init", position_lr_init_, out);
        print_param(" position_lr_final", position_lr_final_, out);
        print_param(" position_lr_delay_mult", position_lr_delay_mult_, out);
        print_param(" position_lr_max_steps", position_lr_max_steps_, out);
        print_param(" feature_lr", feature_lr_, out);
        print_param(" opacity_lr", opacity_lr_, out);
        print_param(" scaling_lr", scaling_lr_, out);
        print_param(" rotation_lr", rotation_lr_, out);
        print_param(" percent_dense", percent_dense_, out);
        print_param(" lambda_dssim", lambda_dssim_, out);
        print_param(" densification_interval", densification_interval_, out);
        print_param(" opacity_reset_interval", opacity_reset_interval_, out);
        print_param(" densify_from_iter", densify_from_iter_, out);
        print_param(" densify_until_iter", densify_until_iter_, out);
        print_param(" densify_grad_threshold", densify_grad_threshold_, out);
        print_param(" random_background", random_background_, out);
    }
};

struct PipelineParams
{
    bool convert_SHs_python_{false};
    bool compute_cov3D_python_{false};
    bool debug_{false};

    void print_params(std::ostream &out = std::cout) const
    {
        out << "> Pipeline parameters:" << std::endl;
        print_param(" convert_SHs_python", convert_SHs_python_, out);
        print_param(" compute_cov3D_python", compute_cov3D_python_, out);
        print_param(" debug", debug_, out);
    }
};

struct OtherParams
{
    std::string ip_{"127.0.0.1"};
    int port_{6009};
    int debug_from_{-1};
    bool detect_anomaly_{false};
    std::vector<int> save_iterations_{7'000, 30'000};
    std::vector<int> test_iterations_{7'000, 30'000};
    bool quiet_{false};
    std::vector<int> checkpoint_iterations_{};
    std::string start_checkpoint_{};

    void print_params(std::ostream &out = std::cout) const
    {
        out << "> Other parameters:" << std::endl;
        print_param(" ip", ip_, out);
        print_param(" port", port_, out);
        print_param(" debug_from", debug_from_, out);
        print_param(" detect_anomaly", detect_anomaly_, out);
        print_param(" save_iterations", save_iterations_, out);
        print_param(" test_iterations", test_iterations_, out);
        print_param(" quiet", quiet_, out);
        print_param(" checkpoint_iterations", checkpoint_iterations_, out);
        print_param(" start_checkpoint", start_checkpoint_, out);
    }
};

auto parse_parameters(int argc, const char *argv[])
    -> boost::optional<
        std::tuple<
            ModelParams, OptimizationParams, PipelineParams, OtherParams>>;