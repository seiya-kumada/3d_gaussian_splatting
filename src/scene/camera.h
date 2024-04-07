#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <optional>

class Camera : public torch::nn::Module
{
private:
    int uid_;
    int colmap_id_;
    cv::Matx33d R_;
    cv::Vec3d T_;
    double FoVx_;
    double FoVy_;
    torch::Tensor image_;
    std::string image_name_;
    cv::Vec3d trans_;
    double scale_;
    torch::TensorOptions data_device_;
    torch::Tensor original_image_;
    int image_width_;
    int image_height_;
    double zfar_;
    double znear_;
    torch::Tensor world_view_transform_;
    torch::Tensor projection_matrix_;
    torch::Tensor full_proj_transform_;
    torch::Tensor camera_center_;

public:
    Camera(
        int uid,
        int colmap_id, 
        const cv::Matx33d &R,
        const cv::Vec3d &T,
        const double &FoVx,
        const double &FoVy,
        const torch::Tensor &image,
        const std::optional<torch::Tensor>& gt_alpha_mask,
        const std::string& image_name,
        const cv::Vec3d& trans= cv::Vec3d{0.0, 0.0, 0.0},
        const double& scale=1.0,
        const std::string& data_device="cuda");

};