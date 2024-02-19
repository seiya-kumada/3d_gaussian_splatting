#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "utils/train_utils.h"
#include <boost/format.hpp>
#include "arguments/params.h"
#include <filesystem>
#include <fstream>
#include "scene/gaussian_model.h"
#include "scene/scene.h"

namespace fs = std::filesystem;

Printer::Printer(bool quiet)
    : quiet_{quiet}
{
}

void Printer::print(const std::string &msg) const
{
    if (quiet_)
    {
        std::cout << boost::format("%1%\n") % msg;
    }
    else
    {
        // 現在の日付と時刻を取得
        boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();

        // 出力
        std::cout << boost::format("%1% [%2%]\n") % msg % now;
    }
};

void initialize_random_seed()
{
    // initialize random seed
    srand(time(NULL));

    // 乱数生成器のシードを設定
    torch::manual_seed(0); // CPUのシードを設定

    if (torch::cuda::is_available())
    {
        // CUDAが利用可能な場合、CUDAのシードも設定
        torch::cuda::manual_seed(0);
        torch::cuda::manual_seed_all(0); // 複数のCUDAデバイスがある場合には、すべてのデバイスに対してシードを設定
    }
}

namespace
{
    void create_directories(const Printer &printer, const ModelParams &model_params)
    {
        printer.print((boost::format("Output folder: %1%") % model_params.model_path_).str());
        if (!fs::exists(model_params.model_path_))
        {
            fs::create_directories(model_params.model_path_);
        }
        else
        {
            // if it does exist, make sure it's empty
            for (auto &p : fs::directory_iterator(model_params.model_path_))
            {
                fs::remove_all(p);
            }
        }
    }

    void save_params(const ModelParams &model_params)
    {
        auto path = fs::path(model_params.model_path_) / "cfg_args";
        auto ofs = std::ofstream{path.string()};
        model_params.print_params(ofs);
    }

    void prepare_output_and_logger(ModelParams &model_params, const Printer &printer)
    {
        if (model_params.model_path_.empty())
        {
            // make a path to the output directory using appropriate name
            auto now = boost::posix_time::second_clock::local_time();
            auto now_str = boost::posix_time::to_iso_string(now);
            model_params.model_path_ = fs::path("./output") / now_str;
        }

        // make the output directory if it doesn't exist
        create_directories(printer, model_params);

        // save the parameters in model_params to the output directory
        save_params(model_params);

        // ignore tensorboard writer!!
    }

}

void train(
    ModelParams &model_params,
    const OptimizationParams &optimization_params,
    const PipelineParams &pipeline_params,
    const OtherParams &other_params,
    const Printer &printer)
{
    auto first_iter = int{0};
    prepare_output_and_logger(model_params, printer);
    auto gaussians = std::make_shared<GaussianModel>(model_params.sh_degree_);
    auto scene = std::make_unique<Scene>(model_params, gaussians);
    gaussians->setup(optimization_params);

    if (!other_params.start_checkpoint_.empty())
    {
        // torch::load(other_params.start_checkpoint_);
    }

    auto bg_color = model_params.white_background_ ? std::vector<float>{1.0, 1.0, 1.0}
                                                   : std::vector<float>{0.0, 0.0, 0.0};
    auto background = torch::tensor(bg_color, torch::kFloat32).to(torch::kCUDA);

    // not supported yet in C++
    // iter_start = torch.cuda.Event(enable_timing = True)
    // iter_end = torch.cuda.Event(enable_timing = True)

    auto viewpoint_stack = std::shared_ptr<Camera>{nullptr};
    auto ema_loss_for_log = float{0.0};

    first_iter += 1;
    // loop between first_iter and optimization_params.iterations_
    for (auto iteration = first_iter; iteration <= optimization_params.iterations_; iteration++)
    {
        gaussians->update_learning_rate(iteration);

        // Every 1000 iterations, we increase the levels of SH up to a maximum degree
        if (iteration % 1000 == 0)
        {
            gaussians->oneup_SH_degree();
        }

        // Pick a random Camera
        if (!viewpoint_stack)
        {
            // copy!
            // TODO: std::optional<Camera>の方が良いかも。
            viewpoint_stack = std::make_shared<Camera>(*scene->get_train_camera());
        }
    }
}

#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(test_train_utils)
{
    BOOST_CHECK_EQUAL(1, 1);
}
#endif // UNIT_TEST
