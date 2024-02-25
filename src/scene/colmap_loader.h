#include <string>
#include <map>
#include <vector>
#include <array>
#include <unordered_map>
#include <opencv4/opencv2/core.hpp>

namespace colmap
{
    struct Image
    {
        int id_;
        // std::array<double, 4> qvec_;
        // std::array<double, 3> tvec_;
        cv::Vec4d qvec_;
        cv::Vec3d tvec_;
        int camera_id_;
        std::string name_;
        std::vector<std::pair<double, double>> xys_;
        std::vector<int64_t> point3D_ids_;

        Image();
        Image(
            int id,
            // const std::array<double, 4> &qvec,
            // const std::array<double, 3> &tvec,
            const cv::Vec4d &qvec,
            const cv::Vec3d &tvec,
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

/**
 * Reads the extrinsics from a binary file and returns a mapping of image IDs to colmap::Image objects.
 *
 * @param path_to_model_file The path to the model file containing the extrinsics.
 * @return A mapping of image IDs to colmap::Image objects.
 */
auto read_extrinsics_binary(const std::string &path_to_model_file)
    -> std::unordered_map<int, colmap::Image>;

/**
 * Reads the intrinsics from a binary file and returns a map of camera IDs to Camera objects.
 *
 * @param path_to_model_file The path to the binary model file.
 * @return A map of camera IDs to Camera objects.
 */
auto read_intrinsics_binary(const std::string &path_to_model_file)
    -> std::unordered_map<int, colmap::Camera>;

/**
 * Reads extrinsics from a text file and returns a map of image IDs to colmap::Image objects.
 *
 * @param path_to_model_file The path to the model file containing extrinsics information.
 * @return A map of image IDs to colmap::Image objects.
 */
auto read_extrinsics_text(const std::string &path_to_model_file)
    -> std::unordered_map<int, colmap::Image>;

/**
 * Reads the intrinsics(camera parameters) from a text file and returns a map of camera IDs to colmap::Camera objects.
 *
 * @param path_to_model_file The path to the model file containing the intrinsics information.
 * @return A map of camera IDs to colmap::Camera objects representing the intrinsics.
 */
auto read_intrinsics_text(const std::string &path_to_model_file)
    -> std::unordered_map<int, colmap::Camera>;

/**
 * @brief Converts a quaternion vector to a rotation matrix.
 *
 * @param qvec The quaternion vector to be converted.
 * @return The resulting rotation matrix.
 */
auto qvec2rotmat(const cv::Vec4d &qvec)
    -> cv::Matx33d;