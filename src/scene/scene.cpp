#include "scene/scene.h"
#include "scene/gaussian_model.h"
#include "arguments/params.h"

Scene::Scene(
    const ModelParams &model_params,
    std::shared_ptr<GaussianModel> model,
    std::optional<int> load_iteration,         // = std::nullopt,
    bool shuffle,                              // = true,
    const std::list<float> &resolution_scales) // = {1.0})
    : model_path_{model_params.model_path_},
      loaded_iter_{std::nullopt},
      model_{model}
{
}

auto Scene::get_train_camera(float scale) -> std::shared_ptr<Camera>
{
    return train_cameras_[scale];
}

auto Scene::get_test_camera(float scale) -> std::shared_ptr<Camera>
{
    return test_cameras_[scale];
}