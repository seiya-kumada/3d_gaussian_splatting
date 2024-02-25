#include <string>
#include <map>
#include <vector>
#include <array>
#include <unordered_map>

namespace colmap
{
    struct Image
    {
        int id_;
        std::array<double, 4> qvec_;
        std::array<double, 3> tvec_;
        int camera_id_;
        std::string name_;
        std::vector<std::pair<double, double>> xys_;
        std::vector<int64_t> point3D_ids_;

        Image();
        Image(
            int id,
            const std::array<double, 4> &qvec,
            const std::array<double, 3> &tvec,
            int camera_id,
            const std::string &name,
            const std::vector<std::pair<double, double>> &xys,
            const std::vector<int64_t> &point3D_ids);
    };

    struct Camera
    {
        int id_;
        std::string model_;
        uint64_t width_;
        uint64_t height_;
        std::vector<double> params_;

        Camera(
            int id,
            const std::string &model,
            uint64_t width,
            uint64_t height,
            const std::vector<double> &params);
        Camera();
    };
}

auto read_extrinsics_binary(const std::string &path_to_model_file) -> std::unordered_map<int, colmap::Image>;
auto read_intrinsics_binary(const std::string &path_to_model_file) -> std::unordered_map<int, colmap::Camera>;
auto read_extrinsics_text(const std::string &path_to_model_file) -> std::unordered_map<int, colmap::Image>;
auto read_intrinsics_text(const std::string &path_to_model_file) -> std::unordered_map<int, colmap::Camera>;
auto qvec2rotmat(const std::array<double, 4> &qvec) -> void;