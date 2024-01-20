#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "src/utils/train_utils.h"
#include <boost/format.hpp>
#include "src/arguments/params.h"
#include <filesystem>
#include <fstream>
#include "src/scene/gaussian_model.h"
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
    // TODO: set up CUDA device
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
    auto gaussian_model = GaussianModel{model_params.sh_degree_};
}

#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(test_train_utils)
{
    BOOST_CHECK_EQUAL(1, 1);
}
#endif // UNIT_TEST
