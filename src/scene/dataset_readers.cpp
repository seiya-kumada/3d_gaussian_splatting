#include "dataset_readers.h"
#include <filesystem>
#include "colmap_loader.h"
#include <boost/range/adaptor/indexed.hpp>
#include <iostream>
#include <boost/format.hpp>

namespace fs = std::filesystem;

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
            extr.qvec_;
            extr.tvec_;
        }

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
    // std::vector<CameraInfo> cam_infos_unsorted; // = readColmapCameras(...);
    //// ソートなどの処理

    // std::vector<CameraInfo> train_cam_infos, test_cam_infos;
    //// train_cam_infos と test_cam_infos の分割処理

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