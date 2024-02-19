#include "colmap_loader.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <variant>
#include <cstring>
Image::Image() {}

Image::Image(
    int id,
    const std::array<double, 4> &qvec,
    const std::array<double, 3> &tvec,
    int camera_id,
    const std::string &name,
    const std::vector<std::pair<double, double>> &xys,
    const std::vector<int64_t> &point3D_ids)
    : id_{id},
      qvec_{qvec},
      tvec_{tvec},
      camera_id_{camera_id},
      name_{name},
      xys_{xys},
      point3D_ids_{point3D_ids}
{
}

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

    template <typename T>
    std::vector<T> sliceEveryThirdElement(const std::vector<T> &input, size_t start, size_t interval)
    {
        std::vector<T> result;
        // 開始位置が入力ベクタのサイズ以上の場合、空のベクタを返す
        if (start >= input.size())
        {
            return result;
        }
        for (size_t i = start; i < input.size(); i += interval)
        {
            result.push_back(input[i]);
        }
        return result;
    }

}

auto read_extrinsics_binary(const std::string &path_to_model_file) -> std::map<uint64_t, Image>
{
    std::ifstream fid(path_to_model_file, std::ios::binary);
    if (!fid.is_open())
    {
        throw std::runtime_error("Unable to open file: " + path_to_model_file);
    }

    std::map<uint64_t, Image> images;
    uint64_t num_reg_images = std::get<uint64_t>(read_next_bytes(fid, 8, "Q")[0]);
    // std::cout << "_/_/_/num_reg_images:" << num_reg_images << std::endl;
    for (uint64_t i = 0; i < num_reg_images; ++i)
    {
        auto binary_image_properties = read_next_bytes(fid, 64, "idddddddi");
        auto image_id = std::get<int>(binary_image_properties[0]);
        auto qvec = std::array<double, 4>{
            std::get<double>(binary_image_properties[1]),
            std::get<double>(binary_image_properties[2]),
            std::get<double>(binary_image_properties[3]),
            std::get<double>(binary_image_properties[4])};

        auto tvec = std::array<double, 3>{
            std::get<double>(binary_image_properties[5]),
            std::get<double>(binary_image_properties[6]),
            std::get<double>(binary_image_properties[7])};
        auto camera_id = std::get<int>(binary_image_properties[8]);
        auto current_char = std::get<char>(read_next_bytes(fid, 1, "c")[0]);
        // std::cout << "_/_/_/image_id:" << image_id << std::endl;
        // std::cout << "_/_/_/camera_id:" << camera_id << std::endl;
        auto image_name = std::string{};
        while (current_char != '\0')
        {
            image_name += current_char;
            current_char = std::get<char>(read_next_bytes(fid, 1, "c")[0]);
        }
        // std::cout << "_/_/_/image_name:" << image_name << std::endl;
        uint64_t num_points2D = std::get<uint64_t>(read_next_bytes(fid, 8, "Q")[0]);
        // std::cout << "_/_/_/num_points2D:" << num_points2D << std::endl;

        std::vector<std::pair<double, double>> xys;
        std::vector<int64_t> point3D_ids;
        for (uint64_t j = 0; j < num_points2D; ++j)
        {
            auto x_y_id_s = read_next_bytes(fid, 24, "ddq");
            double x = std::get<double>(x_y_id_s[0]);
            double y = std::get<double>(x_y_id_s[1]);
            int64_t point_id = std::get<int64_t>(x_y_id_s[2]);
            xys.emplace_back(x, y);
            point3D_ids.push_back(point_id);
        }
        images[image_id] = Image(image_id, qvec, tvec, camera_id, image_name, xys, point3D_ids);
    }
    return images;
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
    }
}

BOOST_AUTO_TEST_CASE(test_colmap_loader)
{
    std::cout << "test_colmap_loader" << std::endl;
    test_read_extrinsics_binary();
}
#endif // UNIT_TEST