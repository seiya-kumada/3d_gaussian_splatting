#include <inttypes.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

auto focal2fov(const double &focal_length_x, const uint64_t &height) -> double;

/**
 * Calculates the world to view transformation matrix.
 *
 * @param R The rotation 3x3 matrix representing the orientation of the camera.
 * @param t The translation 3d vector representing the position of the camera.
 * @return The 4x4 transformation matrix from world coordinates to view coordinates.
 */
auto get_world2view(const cv::Matx33d &R, const cv::Vec3d &t) -> cv::Matx44d;

auto get_world2view_2(
    const cv::Matx33d &R, 
    const cv::Vec3d &t, 
    const cv::Vec3d &translate = cv::Vec3d{0.0, 0.0, 0.0},
    const double& scale = 1.0) -> cv::Matx44d;

auto get_projection_matrix(
    const double &znear,
    const double &zfar,
    const double &fovX,
    const double &fovY) -> torch::Tensor;