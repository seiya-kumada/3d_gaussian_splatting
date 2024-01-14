#include "arguments/params.h"
#include <iostream>
#include <boost/format.hpp>
#include "utils/train_utils/util.h"
#include <torch/csrc/autograd/anomaly_mode.h>

int main(int argc, const char *argv[])
{
    // set up command line argument parser
    auto is_good = parse_parameters(argc, argv);
    if (!is_good)
    {
        return 1;
    }

    // get various parameters
    auto [model_params,
          optimization_params,
          pipeline_params,
          other_params] = is_good.value();
    // model_params.print_params();
    // optimization_params.print_params();
    // pipeline_params.print_params();
    // other_params.print_params();
    other_params.save_iterations_.emplace_back(optimization_params.iterations_);
    std::cout << boost::format("Optimizing %1%\n") % model_params.model_path_;

    // initialize system state
    initialize_random_seed();
    auto printer = Printer{other_params.quiet_};

    // function to detect areas where backpropagation fails.
    torch::autograd::AnomalyMode::set_enabled(other_params.detect_anomaly_);

    // train
    train(model_params, optimization_params, pipeline_params, other_params, printer);

    // all done
    printer.print("Training complete.");
    return 0;
}
