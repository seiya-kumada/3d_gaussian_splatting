#include "colmap_loader.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <variant>
#include <cstring>

colmap::Image::Image() {}

colmap::Image::Image(
    int id,
    // const std::array<double, 4> &qvec,
    // const std::array<double, 3> &tvec,
    const cv::Vec4d &qvec,
    const cv::Vec3d &tvec,
    int camera_id,
    const std::string &name,
    const std::vector<std::pair<double, double>> &xys,
    const std::vector<int64_t> &point3D_ids)
    : id_{id},
      qvec_{qvec},
      tvec_{tvec},
      // qvec_cv_{qvec_cv},
      // tvec_cv_{tvec_cv},
      camera_id_{camera_id},
      name_{name},
      xys_{xys},
      point3D_ids_{point3D_ids}
{
}

colmap::Camera::Camera() {}
colmap::Camera::Camera(
    int id,
    const std::string &model,
    uint64_t width,
    uint64_t height,
    const std::vector<double> &params)
    : id_{id},
      model_{model},
      width_{width},
      height_{height},
      params_{params} {}

namespace
{
    template <typename T>
    T read_data_core(const char *&cursor)
    {
        T value;
        std::memcpy(&value, cursor, sizeof(T));
        cursor += sizeof(T);
        return value;
    }
    // データ型を格納するためのvariant
    using DataVariant = std::variant<int, double, uint64_t, char, int64_t>;

    // フォーマットに従ってデータを読み込む関数
    std::vector<DataVariant> read_data(const char *data, const std::string &format_char_sequence)
    {
        std::vector<DataVariant> result;
        const char *cursor = data;
        for (char fmt : format_char_sequence)
        {
            switch (fmt)
            {
            case 'i':
            { // 整数
                auto value = read_data_core<int>(cursor);
                result.push_back(value);
                break;
            }
            case 'd':
            { // 倍精度浮動小数点数
                auto value = read_data_core<double>(cursor);
                result.push_back(value);
                break;
            }
            case 'Q':
            { // uint64_t
                auto value = read_data_core<uint64_t>(cursor);
                result.push_back(value);
                break;
            }
            case 'c':
            { // char
                auto value = read_data_core<char>(cursor);
                result.push_back(value);
                break;
            }
            case 'q':
            { // long long
                auto value = read_data_core<int64_t>(cursor);
                result.push_back(value);
                break;
            }
            // 他のフォーマット文字に対応する場合は、ここにケースを追加
            default:
                // 未知のフォーマット文字の処理
                std::cerr << "Unsupported format character: " << fmt << std::endl;
                break;
            }
        }

        return result;
    }

    auto read_next_bytes(
        std::ifstream &stream,
        int num_bytes,
        const std::string &format_char_sequence) -> std::vector<DataVariant>
    {
        auto data = std::vector<char>(num_bytes);
        stream.read(data.data(), num_bytes);
        return read_data(data.data(), format_char_sequence);
    }
}

// test passed
auto read_extrinsics_binary(const std::string &path_to_model_file) -> std::unordered_map<int, colmap::Image>
{
    std::ifstream file(path_to_model_file, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + path_to_model_file);
    }

    std::unordered_map<int, colmap::Image> images;
    uint64_t num_reg_images = std::get<uint64_t>(read_next_bytes(file, 8, "Q")[0]);
    for (uint64_t i = 0; i < num_reg_images; ++i)
    {
        auto binary_image_properties = read_next_bytes(file, 64, "idddddddi");
        auto image_id = std::get<int>(binary_image_properties[0]);
        auto qvec = cv::Vec4d{
            std::get<double>(binary_image_properties[1]),
            std::get<double>(binary_image_properties[2]),
            std::get<double>(binary_image_properties[3]),
            std::get<double>(binary_image_properties[4])};

        auto tvec = cv::Vec3d{
            std::get<double>(binary_image_properties[5]),
            std::get<double>(binary_image_properties[6]),
            std::get<double>(binary_image_properties[7])};
        auto camera_id = std::get<int>(binary_image_properties[8]);
        auto current_char = std::get<char>(read_next_bytes(file, 1, "c")[0]);
        auto image_name = std::string{};
        while (current_char != '\0')
        {
            image_name += current_char;
            current_char = std::get<char>(read_next_bytes(file, 1, "c")[0]);
        }
        uint64_t num_points2D = std::get<uint64_t>(read_next_bytes(file, 8, "Q")[0]);

        std::vector<std::pair<double, double>> xys;
        std::vector<int64_t> point3D_ids;
        xys.reserve(num_points2D);
        point3D_ids.reserve(num_points2D);
        for (uint64_t j = 0; j < num_points2D; ++j)
        {
            auto x_y_id_s = read_next_bytes(file, 24, "ddq");
            double x = std::get<double>(x_y_id_s[0]);
            double y = std::get<double>(x_y_id_s[1]);
            int64_t point_id = std::get<int64_t>(x_y_id_s[2]);
            xys.emplace_back(x, y);
            point3D_ids.emplace_back(point_id);
        }
        images[image_id] = colmap::Image(image_id, qvec, tvec, camera_id, image_name, xys, point3D_ids);
    }
    return images;
}

// not implemented
auto read_extrinsics_text(const std::string &path_to_model_file) -> std::unordered_map<int, colmap::Image>
{
    std::ifstream file(path_to_model_file, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + path_to_model_file);
    }

    std::unordered_map<int, colmap::Image> images;
    return images;
}

namespace
{
    struct CameraModel
    {
        int model_id_;
        std::string model_name_;
        int num_params_;
    };

    const std::vector<CameraModel> CAMERA_MODELS{
        {0, "SIMPLE_PINHOLE", 3},
        {1, "PINHOLE", 4},
        {2, "SIMPLE_RADIAL", 4},
        {3, "RADIAL", 5},
        {4, "OPENCV", 8},
        {5, "OPENCV_FISHEYE", 8},
        {6, "FULL_OPENCV", 12},
        {7, "FOV", 5},
        {8, "SIMPLE_RADIAL_FISHEYE", 4},
        {9, "RADIAL_FISHEYE", 5},
        {10, "THIN_PRISM_FISHEYE", 12},
    };

    auto make_camera_model_ids() -> std::unordered_map<int, CameraModel>
    {
        std::unordered_map<int, CameraModel> result;
        for (const auto &model : CAMERA_MODELS)
        {
            result[model.model_id_] = model;
        }
        return result;
    }
    const std::unordered_map<int, CameraModel> CAMERA_MODEL_IDS = make_camera_model_ids();

}

// test passed
auto read_intrinsics_binary(const std::string &path_to_model_file) -> std::unordered_map<int, colmap::Camera>
{
    std::ifstream file(path_to_model_file, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + path_to_model_file);
    }

    std::unordered_map<int, colmap::Camera> cameras;
    uint64_t num_cameras = std::get<uint64_t>(read_next_bytes(file, 8, "Q")[0]);

    for (uint64_t i = 0; i < num_cameras; ++i)
    {
        auto camera_properties = read_next_bytes(file, 24, "iiQQ");
        auto camera_id = std::get<int>(camera_properties[0]);
        auto model_id = std::get<int>(camera_properties[1]);
        auto width = std::get<uint64_t>(camera_properties[2]);
        auto height = std::get<uint64_t>(camera_properties[3]);
        const auto &model = CAMERA_MODEL_IDS.at(model_id);
        std::vector<double> params(model.num_params_);
        for (int p = 0; p < model.num_params_; ++p)
        {
            params[p] = std::get<double>(read_next_bytes(file, 8, "d")[0]);
        }
        cameras[camera_id] = colmap::Camera(camera_id, model.model_name_, width, height, params);
    }
    return cameras;
}

// not implemented
auto read_intrinsics_text(const std::string &path_to_model_file) -> std::unordered_map<int, colmap::Camera>
{
    std::ifstream file(path_to_model_file, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + path_to_model_file);
    }

    std::unordered_map<int, colmap::Camera> cameras;
    return cameras;
}

// test passed
auto qvec2rotmat(const cv::Vec4d &qvec) -> cv::Matx33d
{
    return cv::Matx33d{
        1 - 2 * qvec(2) * qvec(2) - 2 * qvec(3) * qvec(3),
        2 * qvec(1) * qvec(2) - 2 * qvec(0) * qvec(3),
        2 * qvec(3) * qvec(1) + 2 * qvec(0) * qvec(2),

        2 * qvec(1) * qvec(2) + 2 * qvec(0) * qvec(3),
        1 - 2 * qvec(1) * qvec(1) - 2 * qvec(3) * qvec(3),
        2 * qvec(2) * qvec(3) - 2 * qvec(0) * qvec(1),

        2 * qvec(3) * qvec(1) - 2 * qvec(0) * qvec(2),
        2 * qvec(2) * qvec(3) + 2 * qvec(0) * qvec(1),
        1 - 2 * qvec(1) * qvec(1) - 2 * qvec(2) * qvec(2)};
}

#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>
namespace
{
    void test_read_extrinsics_binary()
    {
        std::cout << " test_read_extrinsics_binary" << std::endl;
        std::string path = "/home/ubuntu/data/gaussian_splatting/earth_brain/sparse/0/images.bin";
        auto cam_extrinsics = read_extrinsics_binary(path);
        BOOST_CHECK_EQUAL(138, std::size(cam_extrinsics));
        const auto &image = cam_extrinsics.at(1);
        BOOST_CHECK_EQUAL(1, image.id_);
        BOOST_CHECK_EQUAL(1, image.camera_id_);
        BOOST_CHECK_EQUAL("Image_000001.jpg", image.name_);
        BOOST_CHECK_EQUAL(792, image.xys_.size());
        BOOST_CHECK_EQUAL(792, image.point3D_ids_.size());
    }

    void test_read_intrinsics_binary()
    {
        std::cout << " test_read_intrinsics_binary" << std::endl;
        std::string path = "/home/ubuntu/data/gaussian_splatting/earth_brain/sparse/0/cameras.bin";
        auto cam_intrinsics = read_intrinsics_binary(path);
        BOOST_CHECK_EQUAL(1, std::size(cam_intrinsics));
        const auto &camera = cam_intrinsics.at(1);

        BOOST_CHECK_EQUAL("PINHOLE", camera.model_);
        BOOST_CHECK_EQUAL(512, camera.width_);
        BOOST_CHECK_EQUAL(384, camera.height_);
        BOOST_CHECK_EQUAL(4, std::size(camera.params_));
    }

    void test_qvec2rotmat()
    {
        std::cout << " test_qvec2rotmat" << std::endl;
        auto qvec = cv::Vec4d{1.0, 2.0, 3.0, 4.0};
        auto r = qvec2rotmat(qvec);
        BOOST_CHECK_EQUAL(-49, r(0, 0));
        BOOST_CHECK_EQUAL(4, r(0, 1));
        BOOST_CHECK_EQUAL(22, r(0, 2));
        BOOST_CHECK_EQUAL(20, r(1, 0));
        BOOST_CHECK_EQUAL(-39, r(1, 1));
        BOOST_CHECK_EQUAL(20, r(1, 2));
        BOOST_CHECK_EQUAL(10, r(2, 0));
        BOOST_CHECK_EQUAL(28, r(2, 1));
        BOOST_CHECK_EQUAL(-25, r(2, 2));
    }
}

BOOST_AUTO_TEST_CASE(test_colmap_loader)
{
    std::cout << "test_colmap_loader" << std::endl;
    test_read_extrinsics_binary();
    test_read_intrinsics_binary();
    test_qvec2rotmat();
}
#endif // UNIT_TEST