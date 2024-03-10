#include "system_utils.h"
#include <filesystem>

namespace fs = std::filesystem;

// test passed
auto search_for_max_iteration(std::string &folder) -> int
{
    int maxIter = -1;

    for (const auto &entry : fs::directory_iterator(folder))
    {
        std::string filename = entry.path().filename().string();
        std::size_t pos = filename.find_last_of('_');
        if (pos != std::string::npos)
        {
            std::string iterStr = filename.substr(pos + 1);
            // 数値に変換できるか確認します
            try
            {
                int iter = std::stoi(iterStr);
                maxIter = std::max(maxIter, iter);
            }
            catch (const std::invalid_argument &ia)
            {
                throw std::runtime_error("Invalid iteration number in " + filename);
            }
        }
    }
    return maxIter;
}



#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>
namespace
{
    void test_search_for_max_iteration()
    {
        std::cout << " test_search_for_max_iteration" << std::endl;
        std::string folder = "/home/ubuntu/data/gaussian_splatting/earth_brain/output_2023_10_11";
        auto path = (fs::path(folder) / "point_cloud").string();
        int max_iter = search_for_max_iteration(path);
        BOOST_CHECK_EQUAL(max_iter, 7000);
    }
}

BOOST_AUTO_TEST_CASE(test_system_utils)
{
    std::cout << "test_system_utils" << std::endl;
    test_search_for_max_iteration();
}
#endif // UNIT_TEST