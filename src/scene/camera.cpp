#include "scene/camera.h" 
#include <boost/format.hpp>
#include "utils/graphics_utils.h"
namespace 
{
    torch::Tensor convert_cv_matx44d_to_torch_tensor(const cv::Matx44d& mat) {
        // 4x4のdouble型Tensorを作成します。
        torch::Tensor tensor = torch::empty({4, 4}, torch::kF64);

        // OpenCVのMatx44dからPyTorchのTensorにデータをコピーします。
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                tensor[i][j] = mat(i, j);
            }
        }

        return tensor;
  }
}
Camera::Camera(
    int uid,
    int colmap_id, 
    const cv::Matx33d &R,
    const cv::Vec3d &T,
    const double &FoVx,
    const double &FoVy,
    const torch::Tensor &image,
    const std::optional<torch::Tensor>& gt_alpha_mask,
    const std::string& image_name,
    const cv::Vec3d& trans,
    const double& scale,
    const std::string& data_device)
    : torch::nn::Module(),
      uid_{uid},
      colmap_id_{colmap_id},
      R_{R},
      T_{T},
      FoVx_{FoVx},
      FoVy_{FoVy},
      image_{image},
      image_name_{image_name},
      trans_{trans},
      scale_{scale},
      zfar_{100.0},
      znear_{0.01}
{
    try {
        data_device_ = torch::device(data_device);
    } catch(const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Using default device: cuda" << std::endl;
        std::cerr << boost::format("[Warning] Custom device %1% failed, fallback to default cuda device") % data_device;
        data_device_ = torch::device(std::string{"cuda"});
    }

    original_image_ = image_.clamp(0, 1).to(data_device_);
    image_width_ = original_image_.size(2);
    image_height_ = original_image_.size(1);

    if (!gt_alpha_mask) {
        original_image_ *= gt_alpha_mask.value().to(data_device_);
    } else {
        original_image_ *= torch::ones({1, image_height_, image_width_}, torch::kFloat32).to(data_device_);
    } 

    world_view_transform_ = convert_cv_matx44d_to_torch_tensor(
        get_world2view_2(R, T, trans, scale)
    ).transpose(0, 1).cuda();
    projection_matrix_ = get_projection_matrix(znear_, zfar_, FoVx_, FoVy_).transpose(0,1).cuda();
    full_proj_transform_ = (world_view_transform_.unsqueeze(0).bmm(projection_matrix_.unsqueeze(0))).squeeze(0);
    camera_center_ = world_view_transform_.inverse().index({3, torch::indexing::Slice(0, 3)});
 }