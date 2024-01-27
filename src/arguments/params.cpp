#include "params.h"
#include <iostream>
namespace po = boost::program_options;

namespace
{
    typedef void Fn(boost::program_options::options_description_easy_init &);

    void set_arguments_of_model_params(boost::program_options::options_description_easy_init &a)
    {
        a("help", "produce help message for model parameters");
        a("source_path", po::value<std::string>()->default_value(""));
        a("model_path", po::value<std::string>()->default_value(""));
        a("images", po::value<std::string>()->default_value("images"));
        a("resolution", po::value<int>()->default_value(-1));
        a("white_background", po::bool_switch()->default_value(false), "Activate white background");
        a("data_device", po::value<std::string>()->default_value("cuda"));
        a("eval", po::bool_switch()->default_value(false), "Activate eval");
    }

    void set_arguments_of_optimization_params(boost::program_options::options_description_easy_init &a)
    {
        a("iterations", po::value<int>()->default_value(30'000));
        a("position_lr_init", po::value<float>()->default_value(0.00016));
        a("position_lr_final", po::value<float>()->default_value(0.0000016));
        a("position_lr_delay_mult", po::value<float>()->default_value(0.01));
        a("position_lr_max_steps", po::value<int>()->default_value(30'000));
        a("feature_lr", po::value<float>()->default_value(0.0025));
        a("opacity_lr", po::value<float>()->default_value(0.05));
        a("scaling_lr", po::value<float>()->default_value(0.005));
        a("rotation_lr", po::value<float>()->default_value(0.001));
        a("percent_dense", po::value<float>()->default_value(0.01));
        a("lambda_dssim", po::value<float>()->default_value(0.2));
        a("densification_interval", po::value<int>()->default_value(100));
        a("opacity_reset_interval", po::value<int>()->default_value(3000));
        a("densify_from_iter", po::value<int>()->default_value(500));
        a("densify_until_iter", po::value<int>()->default_value(15'000));
        a("densify_grad_threshold", po::value<float>()->default_value(0.0002));
        a("random_background", po::value<bool>()->default_value(false));
    }

    void set_arguments_of_pipeline_params(boost::program_options::options_description_easy_init &a)
    {
        a("convert_SHs_python", po::value<bool>()->default_value(false));
        a("compute_cov3D_python", po::value<bool>()->default_value(false));
        a("debug", po::value<bool>()->default_value(false));
    }

    void set_arguments_of_other_params(boost::program_options::options_description_easy_init &a)
    {
        a("ip", po::value<std::string>()->default_value("127.0.0.1"));
        a("port", po::value<int>()->default_value(6009));
        a("debug_from", po::value<int>()->default_value(-1));
        a("detect_anomaly", po::bool_switch()->default_value(false), "Activate anomaly detection");
        a("save_iterations",
          po::value<std::vector<int>>()->multitoken()->default_value(std::vector<int>{7000, 30000}, "7000 30000"),
          "Specify iteration numbers to save");
        a("test_iterations",
          po::value<std::vector<int>>()->multitoken()->default_value(std::vector<int>{7000, 30000}, "7000 30000"),
          "Specify iteration numbers to test");
        a("quiet", po::bool_switch()->default_value(false), "Activate quiet mode");
        a("checkpoint_iterations",
          po::value<std::vector<int>>()->multitoken()->default_value(std::vector<int>{}, ""),
          "Specify checkpoint iterations");
        a("start_checkpoint", po::value<std::string>()->default_value(""));
    }

    auto add_options(po::options_description &desc, Fn fn)
        -> void
    {
        auto a = desc.add_options();
        fn(a);
    }

    auto add_model_parameters()
        -> po::options_description
    {
        po::options_description desc("> Model Parameters");
        add_options(desc, set_arguments_of_model_params);
        return desc;
    }

    auto add_optimization_parameters()
        -> po::options_description
    {
        po::options_description desc("> Optimization Parameters");
        add_options(desc, set_arguments_of_optimization_params);
        return desc;
    }

    auto add_pipeline_parameters()
        -> po::options_description
    {
        po::options_description desc("> Pipeline Parameters");
        add_options(desc, set_arguments_of_pipeline_params);
        return desc;
    }

    auto add_other_parameters()
        -> po::options_description
    {
        po::options_description desc("> Other Parameters");
        add_options(desc, set_arguments_of_other_params);
        return desc;
    }

    auto parse_parameters_core(int argc, const char *argv[])
        -> boost::optional<po::variables_map>
    {
        // set up command line argument parser
        auto model_params = add_model_parameters();
        auto optimization_params = add_optimization_parameters();
        auto pipeline_params = add_pipeline_parameters();
        auto other_params = add_other_parameters();

        // assemble all parameters
        po::options_description all{"All Parameters"};
        all.add(model_params).add(optimization_params).add(pipeline_params).add(other_params);

        // parse command line arguments
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << all << "\n";
            return boost::none;
        }
        else
        {
            return vm;
        }
    }

    template <typename T>
    inline const T extract_parameter(
        const std::string &name,
        const std::string &error_message,
        const boost::program_options::variables_map &vm)
    {
        if (vm.count(name))
        {
            return vm[name].as<T>();
        }
        else
        {
            throw std::runtime_error(error_message);
        }
    }

    auto set_model_parameters(po::variables_map const &vm) -> ModelParams
    {
        auto params = ModelParams{};
        params.source_path_ = extract_parameter<std::string>(
            "source_path",
            "invalid source_path",
            vm);
        params.model_path_ = extract_parameter<std::string>(
            "model_path",
            "invalid model_path",
            vm);
        params.images_ = extract_parameter<std::string>(
            "images",
            "invalid images",
            vm);
        params.resolution_ = extract_parameter<int>(
            "resolution",
            "invalid resolution",
            vm);
        params.white_background_ = extract_parameter<bool>(
            "white_background",
            "invalid white_background",
            vm);
        params.data_device_ = extract_parameter<std::string>(
            "data_device",
            "invalid data_device",
            vm);
        params.eval_ = extract_parameter<bool>(
            "eval",
            "invalid eval",
            vm);
        return params;
    }

    auto set_optimization_parameters(po::variables_map const &vm) -> OptimizationParams
    {
        auto params = OptimizationParams{};
        params.iterations_ = extract_parameter<int>(
            "iterations",
            "invalid iterations",
            vm);
        params.position_lr_init_ = extract_parameter<float>(
            "position_lr_init",
            "invalid position_lr_init",
            vm);
        params.position_lr_final_ = extract_parameter<float>(
            "position_lr_final",
            "invalid position_lr_final",
            vm);
        params.position_lr_delay_mult_ = extract_parameter<float>(
            "position_lr_delay_mult",
            "invalid position_lr_delay_mult",
            vm);
        params.position_lr_max_steps_ = extract_parameter<int>(
            "position_lr_max_steps",
            "invalid position_lr_max_steps",
            vm);
        params.feature_lr_ = extract_parameter<float>(
            "feature_lr",
            "invalid feature_lr",
            vm);
        params.opacity_lr_ = extract_parameter<float>(
            "opacity_lr",
            "invalid opacity_lr",
            vm);
        params.scaling_lr_ = extract_parameter<float>(
            "scaling_lr",
            "invalid scaling_lr",
            vm);
        params.rotation_lr_ = extract_parameter<float>(
            "rotation_lr",
            "invalid rotation_lr",
            vm);
        params.percent_dense_ = extract_parameter<float>(
            "percent_dense",
            "invalid percent_dense",
            vm);
        params.lambda_dssim_ = extract_parameter<float>(
            "lambda_dssim",
            "invalid lambda_dssim",
            vm);
        params.densification_interval_ = extract_parameter<int>(
            "densification_interval",
            "invalid densification_interval",
            vm);
        params.opacity_reset_interval_ = extract_parameter<int>(
            "opacity_reset_interval",
            "invalid opacity_reset_interval",
            vm);
        params.densify_from_iter_ = extract_parameter<int>(
            "densify_from_iter",
            "invalid densify_from_iter",
            vm);
        params.densify_until_iter_ = extract_parameter<int>(
            "densify_until_iter",
            "invalid densify_until_iter",
            vm);
        params.densify_grad_threshold_ = extract_parameter<float>(
            "densify_grad_threshold",
            "invalid densify_grad_threshold",
            vm);
        params.random_background_ = extract_parameter<bool>(
            "random_background",
            "invalid random_background",
            vm);

        return params;
    }

    auto set_pipeline_parameters(po::variables_map const &vm) -> PipelineParams
    {
        auto params = PipelineParams{};
        params.convert_SHs_python_ = extract_parameter<bool>(
            "convert_SHs_python",
            "invalid convert_SHs_python",
            vm);
        params.compute_cov3D_python_ = extract_parameter<bool>(
            "compute_cov3D_python",
            "invalid compute_cov3D_python",
            vm);
        params.debug_ = extract_parameter<bool>(
            "debug",
            "invalid debug",
            vm);
        return params;
    }

    auto set_other_parameters(po::variables_map const &vm) -> OtherParams
    {
        auto params = OtherParams{};
        params.ip_ = extract_parameter<std::string>(
            "ip",
            "invalid ip",
            vm);
        params.port_ = extract_parameter<int>(
            "port",
            "invalid port",
            vm);
        params.debug_from_ = extract_parameter<int>(
            "debug_from",
            "invalid debug_from",
            vm);
        params.detect_anomaly_ = extract_parameter<bool>(
            "detect_anomaly",
            "invalid detect_anomaly",
            vm);
        params.save_iterations_ = extract_parameter<std::vector<int>>(
            "save_iterations",
            "invalid save_iterations",
            vm);
        params.test_iterations_ = extract_parameter<std::vector<int>>(
            "test_iterations",
            "invalid test_iterations",
            vm);
        params.quiet_ = extract_parameter<bool>(
            "quiet",
            "invalid quiet",
            vm);
        params.checkpoint_iterations_ = extract_parameter<std::vector<int>>(
            "checkpoint_iterations",
            "invalid checkpoint_iterations",
            vm);
        params.start_checkpoint_ = extract_parameter<std::string>(
            "start_checkpoint",
            "invalid start_checkpoint",
            vm);
        return params;
    }
}

auto parse_parameters(int argc, const char *argv[])
    -> boost::optional<
        std::tuple<
            ModelParams, OptimizationParams, PipelineParams, OtherParams>>
{
    auto is_good = parse_parameters_core(argc, argv);
    if (!is_good)
    {
        return boost::none;
    }

    auto model_params = set_model_parameters(is_good.get());
    auto optimization_params = set_optimization_parameters(is_good.get());
    auto pipleline_params = set_pipeline_parameters(is_good.get());
    auto other_params = set_other_parameters(is_good.get());

    auto r = std::make_tuple(
        model_params, optimization_params, pipleline_params, other_params);
    return r;
}

#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(test_params)
{
    // 後回し
    BOOST_CHECK_EQUAL(1, 1);
}
#endif // UNIT_TEST
