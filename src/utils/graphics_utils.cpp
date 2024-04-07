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

// test passed
auto get_projection_matrix(
    const double &znear,
    const double &zfar,
    const double &fovX,
    const double &fovY) -> torch::Tensor {
    
    auto tanHalfFovY = std::tan((fovY / 2));
    auto tanHalfFovX = std::tan((fovX / 2));

    auto top = tanHalfFovY * znear;
    auto bottom = -top;
    auto right = tanHalfFovX * znear;
    auto left = -right;

    auto P = torch::zeros({4, 4}, torch::kF64);

    auto z_sign = 1.0;

    P[0][0] = 2.0 * znear / (right - left);
    P[1][1] = 2.0 * znear / (top - bottom);
    P[0][2] = (right + left) / (right - left);
    P[1][2] = (top + bottom) / (top - bottom);
    P[3][2] = z_sign;
    P[2][2] = z_sign * zfar / (zfar - znear);
    P[2][3] = -(zfar * znear) / (zfar - znear);
    return P;
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

    void test_get_projection_matrix() {
        std::cout << " test_get_projection_matrix" << std::endl;
        const double pi = std::acos(-1);
        auto znear = 1.0;
        auto zfar = 10;
        auto fovX = pi / 2;
        auto fovY = pi / 2;
        auto p = get_projection_matrix(znear, zfar, fovX, fovY);
        BOOST_CHECK_CLOSE(p[0][0].item<double>(), 1.0, 0.0001);
        BOOST_CHECK_CLOSE(p[1][1].item<double>(), 1.0, 0.0001);
        BOOST_CHECK_EQUAL(p[0][2].item<double>(), 0.0);
        BOOST_CHECK_EQUAL(p[1][2].item<double>(), 0.0);
        BOOST_CHECK_EQUAL(p[3][2].item<double>(), 1.0);
        BOOST_CHECK_CLOSE(p[2][2].item<double>(), 10.0/9, 0.0001);
        BOOST_CHECK_CLOSE(p[2][3].item<double>(), -10.0/9, 0.0001);
    }
}

BOOST_AUTO_TEST_CASE(test_graphical_utils)
{
    std::cout << "test_graphical_utils" << std::endl;
    test_get_world2view();    
    test_get_world2view_2();    
    test_get_projection_matrix();
}
#endif // UNIT_TEST