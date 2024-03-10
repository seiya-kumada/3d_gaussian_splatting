#include "utils/graphics_utils.h"
#include <cmath>

auto focal2fov(const double &focal_length_x, const uint64_t &height) -> double
{
    return 2.0 * std::atan(height / (2.0 * focal_length_x));
}

// test passed
auto get_world2view(const cv::Matx33d &R, const cv::Vec3d &t) -> cv::Matx44d{
    cv::Matx44d Rt(0.0); // 4x4の行列を0で初期化

    // 回転行列Rを転置してRtの左上3x3にコピー
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Rt(i, j) = R(j, i); // 転置を行う
        }
    }

    // 平行移動ベクトルtをRtの最後の列にコピー
    for (int i = 0; i < 3; ++i) {
        Rt(i, 3) = t(i);
    }

    // 同次座標のための値を設定
    Rt(3, 3) = 1.0;

    return Rt; 
}

// test passed
auto get_world2view_2(const cv::Matx33d &R, const cv::Vec3d &t, const cv::Vec3d &translate, const double& scale) -> cv::Matx44d {
    auto Rt = get_world2view(R, t);
    auto C2W = Rt.inv();
    
    auto cam_center = cv::Vec3d{C2W(0, 3), C2W(1, 3), C2W(2, 3)};
    cam_center = (cam_center + translate) * scale;
    for (int i = 0; i < 3; ++i) {
        C2W(i, 3) = cam_center(i);
    }
    Rt = C2W.inv();
    return Rt;
}

#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>

namespace {
    void test_get_world2view()
    {
        std::cout << " test_get_world2view" << std::endl;
        cv::Matx33d R(1.0, 2.0, 0.0,
                      0.0, 1.0, 2.0,
                      0.0, 0.0, 1.0);
        cv::Vec3d t(1.0, 2.0, 3.0);
        cv::Matx44d Rt = get_world2view(R, t);
        BOOST_CHECK_EQUAL(Rt(0, 0), 1.0);
        BOOST_CHECK_EQUAL(Rt(1, 1), 1.0);
        BOOST_CHECK_EQUAL(Rt(2, 2), 1.0);
        BOOST_CHECK_EQUAL(Rt(3, 3), 1.0);
        BOOST_CHECK_EQUAL(Rt(0, 3), 1.0);
        BOOST_CHECK_EQUAL(Rt(1, 0), 2.0);
        BOOST_CHECK_EQUAL(Rt(1, 3), 2.0);
        BOOST_CHECK_EQUAL(Rt(2, 1), 2.0);
        BOOST_CHECK_EQUAL(Rt(2, 3), 3.0);
    }

    void test_get_world2view_2()
    {
        std::cout << " test_get_world2view_2" << std::endl;
        cv::Matx33d R(1.0, 2.0, 0.0,
                      0.0, 1.0, 2.0,
                      0.0, 0.0, 1.0);
        cv::Vec3d t(1.0, 2.0, 3.0);
        cv::Matx44d Rt = get_world2view_2(R, t, cv::Vec3d{1.0, 1.0, 1.0}, 1.0);
        BOOST_CHECK_EQUAL(Rt(0, 0), 1.0);
        BOOST_CHECK_EQUAL(Rt(1, 1), 1.0);
        BOOST_CHECK_EQUAL(Rt(2, 2), 1.0);
        BOOST_CHECK_EQUAL(Rt(3, 3), 1.0);
        
        BOOST_CHECK_EQUAL(Rt(1, 0), 2.0);
        BOOST_CHECK_EQUAL(Rt(1, 3), -1.0);
        
        BOOST_CHECK_EQUAL(Rt(2, 1), 2.0);
        BOOST_CHECK_EQUAL(Rt(3, 3), 1.0);
    }
}

BOOST_AUTO_TEST_CASE(test_graphical_utils)
{
    std::cout << "test_graphical_utils" << std::endl;
    test_get_world2view();    
    test_get_world2view_2();    
}
#endif // UNIT_TEST