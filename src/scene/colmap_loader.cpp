#include "colmap_loader.h"
#include <fstream>
#include <vector>
#include <iostream>

Image::Image() {}

Image::Image(
    uint64_t id,
    const std::array<double, 4> &qvec,
    const std::array<double, 3> &tvec,
    int camera_id,
    const std::string &name,
    const std::vector<std::pair<double, double>> &xys,
    const std::vector<int> &point3D_ids)
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
    T readBinary(std::ifstream &stream)
    {
        T data;
        stream.read(reinterpret_cast<char *>(&data), sizeof(T));
        return data;
    }
}

auto read_extrinsics_binary(const std::string &path_to_model_file) -> std::map<uint64_t, Image>
{
    std::ifstream file(path_to_model_file, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + path_to_model_file);
    }

    std::map<uint64_t, Image> images;
    uint64_t num_reg_images = readBinary<uint64_t>(file);
    std::cout << "num_reg_images: " << num_reg_images << std::endl;
    for (uint64_t i = 0; i < num_reg_images; ++i)
    {
        std::cout << i << std::endl;
        uint64_t image_id = readBinary<uint64_t>(file);
        std::array<double, 4> qvec;
        for (int j = 0; j < 4; ++j)
            qvec[j] = readBinary<double>(file);
        std::array<double, 3> tvec;
        for (int j = 0; j < 3; ++j)
            tvec[j] = readBinary<double>(file);
        int camera_id = readBinary<int>(file);
        std::string image_name;
        char current_char;
        while (true)
        {
            file.read(&current_char, 1);
            if (current_char == '\0')
                break;
            image_name += current_char;
        }
        uint64_t num_points2D = readBinary<uint64_t>(file);
        std::vector<std::pair<double, double>> xys;
        std::vector<int> point3D_ids;
        for (uint64_t j = 0; j < num_points2D; ++j)
        {
            double x = readBinary<double>(file);
            double y = readBinary<double>(file);
            int point_id = readBinary<int>(file);
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