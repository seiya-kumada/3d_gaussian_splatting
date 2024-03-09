#include "scene/scene.h"
#include "scene/gaussian_model.h"
#include "arguments/params.h"
#include "utils/system_utils.h"
#include "scene/dataset_readers.h"
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

        //        self.train_cameras = {}
        //        self.test_cameras = {}
        if (fs::exists(fs::path{model_params.source_path_} / "sparse"))
        {
            // scene_load_type_callbacks["Colmap"](
            //     model_params.source_path_,
            //     model_params.images_,
            //     model_params.eval_,
            //     8);
        }
        else if (fs::exists(fs::path{model_params.source_path_} / "transforms_train.json"))
        {
            std::cout << "Found transforms_train.json file, assuming Blender data set!" << std::endl;
            //            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        }
        else
        {
            assert(false && "Could not recognize scene type!");
        }
        //
        //        if os.path.exists(os.path.join(args.source_path, "sparse")):
        //            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        //        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        //            print("Found transforms_train.json file, assuming Blender data set!")
        //            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        //        else:
        //            assert False, "Could not recognize scene type!"
        //
    }

    auto path = (fs::path(model_params.source_path_) / "sparce").string();
    if (fs::exists(path))
    {
    }
}

auto Scene::get_train_camera(float scale) -> std::shared_ptr<CameraInfo>
{
    return train_cameras_[scale];
}

auto Scene::get_test_camera(float scale) -> std::shared_ptr<CameraInfo>
{
    return test_cameras_[scale];
}