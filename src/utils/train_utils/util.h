#include <string>

void initialize_random_seed();
class Printer
{
private:
    bool quiet_;

public:
    Printer(bool quiet);
    void print(const std::string &msg) const;
};

struct ModelParams;
struct OptimizationParams;
struct PipelineParams;
struct OtherParams;
void train(
    ModelParams &model_params,
    const OptimizationParams &optimization_params,
    const PipelineParams &pipeline_params,
    const OtherParams &other_params,
    const Printer &printer);
