#include <memory>
#include <optional>
#include <list>
#include <map>
#include "scene/camera.h"
class GaussianModel;

struct ModelParams;
class Scene
{
public:
    Scene(
        const ModelParams &model_params,
        std::shared_ptr<GaussianModel> model,
        std::optional<int> load_iteration = std::nullopt,
        bool shuffle = true,
        const std::list<float> &resolution_scales = {1.0});

private:
    std::string model_path_;
    std::optional<int> loaded_iter_;
    std::shared_ptr<GaussianModel> model_;

    std::map<float, std::unique_ptr<Camera>> train_cameras_;
    std::map<float, std::unique_ptr<Camera>> test_cameras_;
};