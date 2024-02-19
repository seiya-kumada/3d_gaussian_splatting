#include "dataset_readers.h"
#include <filesystem>
#include "colmap_loader.h"
namespace fs = std::filesystem;

auto read_colmap_scene_info(const std::string &path, const std::string &images, bool eval, int llffhold) -> void
{
    auto cameras_extrinsic_file = std::string{};
    auto cameras_intrinsic_file = std::string{};
    auto cam_extrinsics = std::map<uint64_t, Image>{};
    auto cam_intrinsics = std::map<uint64_t, Image>{};
    try
    {
        cameras_extrinsic_file = fs::path(path) / "sparse" / "0" / "images.bin";
        cameras_intrinsic_file = fs::path(path) / "sparse" / "0" / "cameras.bin";
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file);
        //  cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file);
    }
    catch (...)
    {
        cameras_extrinsic_file = fs::path(path) / "sparse/0" / "images.txt";
        cameras_intrinsic_file = fs::path(path) / "sparse/0" / "cameras.txt";
        // cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file);
        //  cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file);
    }
}
/*
def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info*/
