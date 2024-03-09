#include <string>
#include <opencv2/core.hpp>
#include <unordered_map>

struct CameraInfo
{
    int uid_;
    cv::Matx33d R_;
    cv::Vec3d T_;
    double FovY_;
    double FovX_;
    cv::Mat image_;
    std::string image_path_;
    std::string image_name_;
    int width_;
    int height_;

    CameraInfo(
        int uid,
        const cv::Matx33d &R,
        const cv::Vec3d &T,
        const double &FovY,
        const double &FovX,
        const cv::Mat &image,
        const std::string &image_path,
        const std::string &image_name,
        const int &width,
        const int &height);
};
auto read_colmap_scene_info(const std::string &path, const std::string &images, bool eval, int llffhold = 8) -> void;
auto read_nerf_synthetic_info(const std::string &path, const std::string &images, bool eval, int llffhold = 8) -> void;

// std::unordered_map<
//     std::string,
//     std::function<void(const std::string &, const std::string &, bool, int)>>
//     scene_load_type_callbacks{
//         {"Colmap", read_colmap_scene_info},
//         {"Blender", read_nerf_synthetic_info}};