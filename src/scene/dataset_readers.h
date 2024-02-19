#include <string>
struct CameraInfo
{
    std::string image_name;
    // その他のカメラ情報
};
auto read_colmap_scene_info(const std::string &path, const std::string &images, bool eval, int llffhold = 8) -> void;
// auto read_colmap_camera(const std::string &path, const std::string &images, bool eval, int llffhold = 8) -> void;