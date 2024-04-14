#include "dataset_readers.h"
#include <filesystem>
#include "colmap_loader.h"
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/filtered.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <iostream>
#include <boost/format.hpp>
#include "utils/graphics_utils.h"
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

CameraInfo::CameraInfo(
    int uid,
    const cv::Matx33d &R,
    const cv::Vec3d &T,
    const double &FovY,
    const double &FovX,
    const cv::Mat &image,
    const std::string &image_path,
    const std::string &image_name,
    const int &width,
    const int &height)
    : uid_{uid},
      R_{R},
      T_{T},
      FovY_{FovY},
      FovX_{FovX},
      image_{image},
      image_path_{image_path},
      image_name_{image_name},
      width_{width},
      height_{height}
{
}

namespace
{
    auto read_colmap_cameras(
        const std::unordered_map<int, colmap::Image> &cam_extrinsics,
        const std::unordered_map<int, colmap::Camera> &cam_intrinsics,
        const std::string &images_folder) -> std::vector<CameraInfo>
    {
        std::vector<CameraInfo> cam_infos;
        auto camera = colmap::Camera{};
        auto key = int{};
        for (const auto &p : cam_extrinsics | boost::adaptors::indexed(0))
        {
            std::cout << std::endl;
            std::cout << boost::format("Reading camera %1%/%2%") % (p.index() + 1) % std::size(cam_extrinsics);
            std::cout << std::flush;

            const auto &v = p.value();
            const auto &camera_id = v.first;
            const auto &extr = v.second;
            const auto &intr = cam_intrinsics.at(extr.camera_id_);
            const auto height = intr.height_;
            const auto width = intr.width_;

            const auto uid = intr.id_;
            const auto R = qvec2rotmat(extr.qvec_).t();
            const auto T = extr.tvec_;

            double FovY;
            double FovX;

            if (intr.model_ == "SIMPLE_PINHOLE")
            {
                const auto focal_length_x = intr.params_[0];
                FovY = focal2fov(focal_length_x, height);
                FovX = focal2fov(focal_length_x, width);
            }
            else if (intr.model_ == "PINHOLE")
            {
                const auto focal_length_x = intr.params_[0];
                const auto focal_length_y = intr.params_[1];
                FovY = focal2fov(focal_length_y, height);
                FovX = focal2fov(focal_length_x, width);
            }
            else
            {
                throw std::runtime_error("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!");
            }
            const auto image_path = fs::path(images_folder) / fs::path(extr.name_).filename();
            const auto image_name = image_path.filename().stem();
            const auto image = cv::imread(image_path.string());
            // BGRからRGBへ変換
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

            cam_infos.emplace_back(uid, R, T, FovY, FovX, image, image_path, image_name, width, height);
        }
        std::cout << std::endl;
        return cam_infos;
    }
}

namespace {

    // test passed
    auto get_center_and_diag(const std::vector<cv::Vec3d>& cam_centers) -> std::pair<cv::Vec3d, double> {
        // calculate the sum of all camera centers
        cv::Vec3d sum(0, 0, 0);
        for (const auto& center : cam_centers) {
            sum += center;
        }
        // calculate the average of all camera centers
        cv::Vec3d avg_cam_center = sum / static_cast<double>(cam_centers.size());

        // calculate the maximum distance between each camera center and the average camera center
        double max_dist = 0.0;
        for (const auto& center : cam_centers) {
            double dist = cv::norm(center - avg_cam_center);
            if (dist > max_dist) {
                max_dist = dist;
            }
        }

        return {avg_cam_center, max_dist};
}

    auto get_nerfpp_norm(const std::vector<CameraInfo> &train_cam_infos) 
        -> std::unordered_map<std::string, cv::Vec3d> 
    {
        auto cam_centers = std::vector<cv::Vec3d>{};
        cam_centers.reserve(train_cam_infos.size());
        for (const auto& cam : train_cam_infos) {
            const auto W2C = get_world2view_2(cam.R_, cam.T_);
            const auto C2W = W2C.inv();
            cam_centers.emplace_back(C2W(0, 3), C2W(1, 3), C2W(2, 3));
        }

        auto [center, diagonal] = get_center_and_diag(cam_centers);
        auto radius = diagonal * 1.1;
        auto translate = -center;
        return {{"translate", translate}, {"radius", radius}};
    }
}

auto read_colmap_scene_info(const std::string &path, const std::string &images, bool eval, int llffhold) -> void
{
    auto cameras_extrinsic_file = std::string{};
    auto cameras_intrinsic_file = std::string{};
    auto cam_extrinsics = std::unordered_map<int, colmap::Image>{};
    auto cam_intrinsics = std::unordered_map<int, colmap::Camera>{};
    try
    {
        cameras_extrinsic_file = fs::path(path) / "sparse" / "0" / "images.bin";
        cameras_intrinsic_file = fs::path(path) / "sparse" / "0" / "cameras.bin";
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file);
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file);
    }
    catch (...)
    {
        cameras_extrinsic_file = fs::path(path) / "sparse/0" / "images.txt";
        cameras_intrinsic_file = fs::path(path) / "sparse/0" / "cameras.txt";
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file);
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file);
        throw std::runtime_error("Not implemented");
    }
    std::string reading_dir = !images.empty() ? images : "images";
    std::vector<CameraInfo> cam_infos_unsorted = read_colmap_cameras(cam_extrinsics, cam_intrinsics, fs::path(path) / reading_dir);

    // copy cam_info_unsorted to cam_infos and sort it
    auto cam_infos = cam_infos_unsorted;
    std::sort(cam_infos.begin(), cam_infos.end(), [](const CameraInfo &a, const CameraInfo &b)
              { return a.image_name_ < b.image_name_; });

    std::vector<CameraInfo> train_cam_infos{};
    std::vector<CameraInfo> test_cam_infos{};
    if (eval)
    {
        auto train_tmp = cam_infos |
            boost::adaptors::indexed(0) |
            boost::adaptors::filtered([llffhold](const auto &p) { 
                return p.index() % llffhold != 0; }) |
            boost::adaptors::transformed([](const auto &p) { 
                return p.value(); });
        // convert to std::vector
        train_cam_infos = std::vector<CameraInfo>(boost::begin(train_tmp), boost::end(train_tmp));

        auto test_tmp = cam_infos | 
            boost::adaptors::indexed(0) | 
            boost::adaptors::filtered([llffhold](const auto &p) { 
                return p.index() % llffhold == 0; }) |
            boost::adaptors::transformed([](const auto &p) { 
                return p.value(); });
        test_cam_infos = std::vector<CameraInfo>(boost::begin(test_tmp), boost::end(test_tmp));
    }
    else
    {
        train_cam_infos = cam_infos;
        // test_cam_infos is empty.
    }

    get_nerfpp_norm(train_cam_infos);

    // fs::path ply_path = fs::path(path) / "sparse/0/points3D.ply";
    // fs::path bin_path = fs::path(path) / "sparse/0/points3D.bin";
    // fs::path txt_path = fs::path(path) / "sparse/0/points3D.txt";
    // if (!fs::exists(ply_path)) {
    //     std::cout << "Converting point3d.bin to .ply, will happen only the first time you open the scene." << std::endl;
    //     try {
    //         // xyz, rgb, _ = read_points3D_binary(bin_path);
    //     } catch (...) {
    //         // xyz, rgb, _ = read_points3D_text(txt_path);
    //     }
    //     // storePly(ply_path, xyz, rgb);
    // }

    // std::optional<YourPointCloudType> pcd;
    // try {
    //     // pcd = fetchPly(ply_path);
    // } catch (...) {
    //     pcd = std::nullopt;
    // }

    // SceneInfo scene_info{/* point_cloud= */pcd, /* train_cameras= */train_cam_infos, /* test_cameras= */test_cam_infos/*, その他の情報 */};
    // return scene_info;
}

// TODO
auto read_nerf_synthetic_info(
    const std::string &path,
    const std::string &images,
    bool eval,
    int llffhold) -> void
{
}

#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>
namespace
{
    void test_get_center_and_diag()
    {
        std::cout << " test_get_center_and_diag" << std::endl;
        std::vector<cv::Vec3d> cam_centers = {
            cv::Vec3d{0, 0, 0},
            cv::Vec3d{1, 1, 1},
            cv::Vec3d{2, 2, 2},
            cv::Vec3d{3, 3, 3},
            cv::Vec3d{4, 4, 4},
            cv::Vec3d{5, 5, 5},
            cv::Vec3d{6, 6, 6},
            cv::Vec3d{7, 7, 7},
            cv::Vec3d{8, 8, 8},
            cv::Vec3d{9, 9, 9},
        };
        auto [center, diag] = get_center_and_diag(cam_centers);
        BOOST_CHECK_EQUAL(center[0], 4.5);
        BOOST_CHECK_EQUAL(center[1], 4.5);
        BOOST_CHECK_EQUAL(center[2], 4.5);
        BOOST_CHECK_CLOSE(diag, std::sqrt(3.0 * 4.5 * 4.5), 0.0001);
    }
}

BOOST_AUTO_TEST_CASE(test_dataset_readers)
{
    std::cout << "test_dataset_readers" << std::endl;
    test_get_center_and_diag();
}
#endif // UNIT_TEST