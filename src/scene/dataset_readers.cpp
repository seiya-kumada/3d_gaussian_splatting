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
#include <CppLinq/cpplinq.hpp>

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

    std::vector<CameraInfo> train_cam_infos {};
    std::vector<CameraInfo> test_cam_infos {};
    if (eval)
    {
        auto train_tmp = cam_infos 
            | boost::adaptors::indexed(0) 
            | boost::adaptors::filtered([llffhold](const auto &p) { return p.index() % llffhold != 0; }) 
            | boost::adaptors::transformed([](const auto &p) { return p.value(); });
        // convert to std::vector 
        train_cam_infos = std::vector<CameraInfo>(boost::begin(train_tmp), boost::end(train_tmp));

        auto test_tmp = cam_infos
            | boost::adaptors::indexed(0)
            | boost::adaptors::filtered([llffhold](const auto &p) { return p.index() % llffhold == 0; })
            | boost::adaptors::transformed([](const auto &p) { return p.value(); });
        test_cam_infos = std::vector<CameraInfo> (boost::begin(test_tmp), boost::end(test_tmp));
    }else {
        train_cam_infos = cam_infos;
    }
    
    // next time, you start from here 

    //// nerf_normalization = getNerfppNorm(train_cam_infos);

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

auto read_nerf_synthetic_info(
    const std::string &path,
    const std::string &images,
    bool eval,
    int llffhold) -> void
{
}