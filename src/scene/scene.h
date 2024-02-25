#include <memory>
#include <optional>
#include <list>
#include <map>
#include "scene/camera.h"
class GaussianModel;
class CameraInfo;
struct ModelParams;
class Scene
{
public:
    Scene(
        const ModelParams &model_params,
        std::shared_ptr<GaussianModel> gaussians,
        std::optional<int> load_iteration = std::nullopt,
        bool shuffle = true,
        const std::list<float> &resolution_scales = {1.0});

    auto get_train_camera(float scale = 1.0) -> std::shared_ptr<CameraInfo>;
    auto get_test_camera(float scale = 1.0) -> std::shared_ptr<CameraInfo>;

private:
    std::string model_path_;
    std::optional<int> loaded_iter_;
    std::shared_ptr<GaussianModel> gaussians_;

    std::map<float, std::shared_ptr<CameraInfo>> train_cameras_;
    std::map<float, std::shared_ptr<CameraInfo>> test_cameras_;
};