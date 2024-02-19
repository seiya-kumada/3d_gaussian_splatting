#include <string>
#include <map>
#include <vector>
#include <array>

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

auto read_extrinsics_binary(const std::string &path_to_model_file) -> std::map<uint64_t, Image>;