#include "scene/scene.h"
#include "scene/gaussian_model.h"
#include "arguments/params.h"
#include "utils/system_utils.h"
#include <filesystem>
#include <boost/format.hpp>
namespace fs = std::filesystem;

Scene::Scene(
    const ModelParams &model_params,
    std::shared_ptr<GaussianModel> gaussians,
    std::optional<int> load_iteration,         // = std::nullopt,
    bool shuffle,                              // = true,
    const std::list<float> &resolution_scales) // = {1.0})
    : model_path_{model_params.model_path_},
      loaded_iter_{std::nullopt},
      gaussians_{gaussians}
{
    if (load_iteration)
    {
        if (load_iteration.value() == -1)
        {
            auto path = (fs::path(model_path_) / "point_cloud").string();
            loaded_iter_ = search_for_max_iteration(path);
        }
        else
        {
            loaded_iter_ = load_iteration;
        }

        std::cout << (boost::format("Loading trained model at iteration %1%\n") % loaded_iter_.value());
    }

    auto path = (fs::path(model_params.source_path_) / "sparce").string();
    if (fs::exists(path))
    {
        }
}

auto Scene::get_train_camera(float scale) -> std::shared_ptr<Camera>
{
    return train_cameras_[scale];
}

auto Scene::get_test_camera(float scale) -> std::shared_ptr<Camera>
{
    return test_cameras_[scale];
}