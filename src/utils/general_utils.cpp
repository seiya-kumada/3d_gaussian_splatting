#include "src/utils/general_utils.h"

namespace
{
    // test passed
    /**
     * Builds a rotation matrix from the given quaternion.
     *
     * @param r The quaternion representing the rotation. Its shape is (N, 4).
     * @return The rotation matrix whose shape is (N, 3, 3).
     */
    auto build_rotation(const torch::Tensor &r) -> torch::Tensor
    {
        auto norm = torch::sqrt(r.select(1, 0) * r.select(1, 0) +
                                r.select(1, 1) * r.select(1, 1) +
                                r.select(1, 2) * r.select(1, 2) +
                                r.select(1, 3) * r.select(1, 3));

        auto q = r / norm.unsqueeze(1);
        // q.sizes(): [N, 4]
        auto device = torch::kCUDA;
        torch::Tensor R = torch::zeros({q.size(0), 3, 3}, torch::TensorOptions().device(device));
        // R.sizes(): [N, 3, 3]
        auto rr = q.select(1, 0);
        auto x = q.select(1, 1);
        auto y = q.select(1, 2);
        auto z = q.select(1, 3);
        // rr.size(): [N]
        R.index_put_({torch::indexing::Slice(), 0, 0}, 1 - 2 * (y * y + z * z));
        R.index_put_({torch::indexing::Slice(), 0, 1}, 2 * (x * y - rr * z));
        R.index_put_({torch::indexing::Slice(), 0, 2}, 2 * (x * z + rr * y));
        R.index_put_({torch::indexing::Slice(), 1, 0}, 2 * (x * y + rr * z));
        R.index_put_({torch::indexing::Slice(), 1, 1}, 1 - 2 * (x * x + z * z));
        R.index_put_({torch::indexing::Slice(), 1, 2}, 2 * (y * z - rr * x));
        R.index_put_({torch::indexing::Slice(), 2, 0}, 2 * (x * z - rr * y));
        R.index_put_({torch::indexing::Slice(), 2, 1}, 2 * (y * z + rr * x));
        R.index_put_({torch::indexing::Slice(), 2, 2}, 1 - 2 * (x * x + y * y));

        return R;
    }

    // test passed
    /**
     * @brief
     *
     * @param L The input tensor whose shape is (N, 3, 3).
     * @return The tensor whose shape is (N, 6).
     */
    torch::Tensor strip_lowerdiag(const torch::Tensor &L)
    {
        auto device = torch::kCUDA;
        torch::Tensor uncertainty = torch::zeros({L.size(0), 6}, torch::TensorOptions().dtype(torch::kFloat).device(device));

        uncertainty.select(1, 0) = L.select(1, 0).select(1, 0);
        uncertainty.select(1, 1) = L.select(1, 0).select(1, 1);
        uncertainty.select(1, 2) = L.select(1, 0).select(1, 2);
        uncertainty.select(1, 3) = L.select(1, 1).select(1, 1);
        uncertainty.select(1, 4) = L.select(1, 1).select(1, 2);
        uncertainty.select(1, 5) = L.select(1, 2).select(1, 2);

        return uncertainty;
    }

}

// test passed
/**
 * @brief Strips the symmetric part of a tensor.
 *
 * @param sym The input tensor whose shape is (N, 3, 3).
 * @return The tensor whose shape is (N, 6).
 */
torch::Tensor strip_symmetric(const torch::Tensor &sym)
{
    return strip_lowerdiag(sym);
}

// test passed
/**
 * @brief Builds a scaling and rotation matrix.
 *
 * This function takes two tensors, `s` and `r`, and returns a tensor representing the scaling and rotation matrix.
 *
 * @param s The scaling tensor whose shape is (N, 3).
 * @param r The rotation tensor whose shape is (N, 4).
 * @return The scaling and rotation matrix tensor whose shape is (N, 3, 3).
 */
auto build_scaling_rotation(const torch::Tensor &s, const torch::Tensor &r) -> torch::Tensor
{
    auto device = torch::kCUDA;
    torch::Tensor L = torch::zeros({s.size(0), 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device));
    torch::Tensor R = build_rotation(r); // R.sizes(): [N, 3, 3]

    L.index_put_({torch::indexing::Slice(), 0, 0}, s.select(1, 0));
    L.index_put_({torch::indexing::Slice(), 1, 1}, s.select(1, 1));
    L.index_put_({torch::indexing::Slice(), 2, 2}, s.select(1, 2));
    L = torch::matmul(R, L);
    return L;
}

// test passed
/**
 * @brief Builds the function which returns the learning rate at a given step.
 *
 * @param lr_init The initial learning rate.
 * @param lr_final The final learning rate.
 * @param lr_delay_steps
 * @param lr_delay_mult
 * @param max_steps The number of steps during optimization.
 * @return std::function<float(int)>
 */
auto get_expon_lr_func(
    float lr_init,
    float lr_final,
    int lr_delay_steps,
    float lr_delay_mult,
    int max_steps) -> std::function<float(int)>
{
    return [lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps](int step) -> float
    {
        if (step < 0 || (lr_init == 0.0f && lr_final == 0.0f))
        {
            return 0.0f;
        }

        float delay_rate;
        if (lr_delay_steps > 0)
        {
            // A kind of reverse cosine decay.
            delay_rate = lr_delay_mult +
                         (1 - lr_delay_mult) * std::sin(0.5 * M_PI * std::clamp(static_cast<float>(step) / lr_delay_steps, 0.0f, 1.0f));
        }
        else
        {
            delay_rate = 1.0f;
        }

        float t = std::clamp(static_cast<float>(step) / max_steps, 0.0f, 1.0f);
        float log_lerp = std::exp(std::log(lr_init) * (1 - t) + std::log(lr_final) * t);
        return delay_rate * log_lerp;
    };
}
#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>
namespace
{
    void test_build_rotation()
    {
        std::cout << " test_build_rotation" << std::endl;
        auto device = torch::kCUDA;
        // r = (0.5, 0.5, 0.5, 0.5)
        torch::Tensor r = torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kFloat).device(device));
        r.index_put_({0, 0}, 0.5);
        r.index_put_({0, 1}, 0.5);
        r.index_put_({0, 2}, 0.5);
        r.index_put_({0, 3}, 0.5);
        r.index_put_({1, 0}, 0.25);
        r.index_put_({1, 1}, 0.25);
        r.index_put_({1, 2}, 0.25);
        r.index_put_({1, 3}, 0.25);
        auto R = build_rotation(r);
        BOOST_CHECK_EQUAL(R.size(0), 2);
        BOOST_CHECK_EQUAL(R.size(1), 3);
        BOOST_CHECK_EQUAL(R.size(2), 3);
        BOOST_CHECK_EQUAL(R.dtype(), torch::kFloat);
        BOOST_CHECK_EQUAL(R.device().type(), torch::kCUDA);

        BOOST_CHECK_EQUAL(R.index({0, 0, 0}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(R.index({0, 0, 1}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(R.index({0, 0, 2}).item<float>(), 1.0);
        BOOST_CHECK_EQUAL(R.index({0, 1, 0}).item<float>(), 1.0);
        BOOST_CHECK_EQUAL(R.index({0, 1, 1}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(R.index({0, 1, 2}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(R.index({0, 2, 0}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(R.index({0, 2, 1}).item<float>(), 1.0);
        BOOST_CHECK_EQUAL(R.index({0, 2, 2}).item<float>(), 0.0);

        BOOST_CHECK_EQUAL(R.index({1, 0, 0}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(R.index({1, 0, 1}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(R.index({1, 0, 2}).item<float>(), 1.0);
        BOOST_CHECK_EQUAL(R.index({1, 1, 0}).item<float>(), 1.0);
        BOOST_CHECK_EQUAL(R.index({1, 1, 1}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(R.index({1, 1, 2}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(R.index({1, 2, 0}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(R.index({1, 2, 1}).item<float>(), 1.0);
        BOOST_CHECK_EQUAL(R.index({1, 2, 2}).item<float>(), 0.0);
    }

    void test_build_scaling_rotation()
    {
        std::cout << " test_build_scaling_rotation" << std::endl;
        auto device = torch::kCUDA;
        // s = (0.5, 0.5, 0.5)
        torch::Tensor s = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device));
        s.index_put_({0, 0}, 0.5);
        s.index_put_({0, 1}, 0.5);
        s.index_put_({0, 2}, 0.5);
        s.index_put_({1, 0}, 0.25);
        s.index_put_({1, 1}, 0.25);
        s.index_put_({1, 2}, 0.25);
        // r = (0.5, 0.5, 0.5, 0.5)
        torch::Tensor r = torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kFloat).device(device));
        r.index_put_({0, 0}, 0.5);
        r.index_put_({0, 1}, 0.5);
        r.index_put_({0, 2}, 0.5);
        r.index_put_({0, 3}, 0.5);
        r.index_put_({1, 0}, 0.25);
        r.index_put_({1, 1}, 0.25);
        r.index_put_({1, 2}, 0.25);
        r.index_put_({1, 3}, 0.25);
        auto L = build_scaling_rotation(s, r);
        BOOST_CHECK_EQUAL(L.size(0), 2);
        BOOST_CHECK_EQUAL(L.size(1), 3);
        BOOST_CHECK_EQUAL(L.size(2), 3);
        BOOST_CHECK_EQUAL(L.dtype(), torch::kFloat);
        BOOST_CHECK_EQUAL(L.device().type(), torch::kCUDA);

        BOOST_CHECK_EQUAL(L.index({0, 0, 0}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(L.index({0, 0, 1}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(L.index({0, 0, 2}).item<float>(), 0.5);

        BOOST_CHECK_EQUAL(L.index({0, 1, 0}).item<float>(), 0.5);
        BOOST_CHECK_EQUAL(L.index({0, 1, 1}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(L.index({0, 1, 2}).item<float>(), 0.0);

        BOOST_CHECK_EQUAL(L.index({0, 2, 0}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(L.index({0, 2, 1}).item<float>(), 0.5);
        BOOST_CHECK_EQUAL(L.index({0, 2, 2}).item<float>(), 0.0);

        BOOST_CHECK_EQUAL(L.index({1, 0, 0}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(L.index({1, 0, 1}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(L.index({1, 0, 2}).item<float>(), 0.25);

        BOOST_CHECK_EQUAL(L.index({1, 1, 0}).item<float>(), 0.25);
        BOOST_CHECK_EQUAL(L.index({1, 1, 1}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(L.index({1, 1, 2}).item<float>(), 0.0);

        BOOST_CHECK_EQUAL(L.index({1, 2, 0}).item<float>(), 0.0);
        BOOST_CHECK_EQUAL(L.index({1, 2, 1}).item<float>(), 0.25);
        BOOST_CHECK_EQUAL(L.index({1, 2, 2}).item<float>(), 0.0);
    }

    void test_strip_lowerdiag()
    {
        std::cout << " test_strip_lowerdiag" << std::endl;
        auto device = torch::kCUDA;

        torch::Tensor L = torch::zeros({2, 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device));
        L.index_put_({0, 0, 0}, 1);
        L.index_put_({0, 0, 1}, 2);
        L.index_put_({0, 0, 2}, 3);

        L.index_put_({0, 1, 0}, 4);
        L.index_put_({0, 1, 1}, 5);
        L.index_put_({0, 1, 2}, 6);

        L.index_put_({0, 2, 0}, 7);
        L.index_put_({0, 2, 1}, 8);
        L.index_put_({0, 2, 2}, 9);

        L.index_put_({1, 0, 0}, 10);
        L.index_put_({1, 0, 1}, 20);
        L.index_put_({1, 0, 2}, 30);

        L.index_put_({1, 1, 0}, 40);
        L.index_put_({1, 1, 1}, 50);
        L.index_put_({1, 1, 2}, 60);

        L.index_put_({1, 2, 0}, 70);
        L.index_put_({1, 2, 1}, 80);
        L.index_put_({1, 2, 2}, 90);

        auto uncertainty = strip_lowerdiag(L);
        BOOST_CHECK_EQUAL(uncertainty.size(0), 2);
        BOOST_CHECK_EQUAL(uncertainty.size(1), 6);
        BOOST_CHECK_EQUAL(uncertainty.dtype(), torch::kFloat);
        BOOST_CHECK_EQUAL(uncertainty.device().type(), torch::kCUDA);

        BOOST_CHECK_EQUAL(uncertainty.index({0, 0}).item<float>(), 1);
        BOOST_CHECK_EQUAL(uncertainty.index({0, 1}).item<float>(), 2);
        BOOST_CHECK_EQUAL(uncertainty.index({0, 2}).item<float>(), 3);
        BOOST_CHECK_EQUAL(uncertainty.index({0, 3}).item<float>(), 5);
        BOOST_CHECK_EQUAL(uncertainty.index({0, 4}).item<float>(), 6);
        BOOST_CHECK_EQUAL(uncertainty.index({0, 5}).item<float>(), 9);

        BOOST_CHECK_EQUAL(uncertainty.index({1, 0}).item<float>(), 10);
        BOOST_CHECK_EQUAL(uncertainty.index({1, 1}).item<float>(), 20);
        BOOST_CHECK_EQUAL(uncertainty.index({1, 2}).item<float>(), 30);
        BOOST_CHECK_EQUAL(uncertainty.index({1, 3}).item<float>(), 50);
        BOOST_CHECK_EQUAL(uncertainty.index({1, 4}).item<float>(), 60);
        BOOST_CHECK_EQUAL(uncertainty.index({1, 5}).item<float>(), 90);
    }

    void test_get_expon_lr_func()
    {
        std::cout << " test_get_expon_lr_func" << std::endl;
        {
            auto lr_init = 0.1f;
            auto lr_final = 0.01f;
            auto lr_delay_steps = 100;
            auto lr_delay_mult = 0.5f;
            auto max_steps = 1000;
            auto lr_func = get_expon_lr_func(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps);
            BOOST_CHECK_EQUAL(lr_func(-1), 0.0f);
        }
        {
            auto lr_init = 0.0f;
            auto lr_final = 0.0f;
            auto lr_delay_steps = 100;
            auto lr_delay_mult = 0.5f;
            auto max_steps = 1000;
            auto lr_func = get_expon_lr_func(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps);
            BOOST_CHECK_EQUAL(lr_func(1), 0.0f);
        }
        {
            auto lr_init = 0.1f;
            auto lr_final = 0.01f;
            auto lr_delay_steps = 100;
            auto lr_delay_mult = 0.5f;
            auto max_steps = 1000;
            auto lr_func = get_expon_lr_func(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps);
            BOOST_CHECK_CLOSE(lr_func(0), 0.05f, 1e-5);
        }
        {
            auto lr_init = 0.1f;
            auto lr_final = 0.01f;
            auto lr_delay_steps = 0;
            auto lr_delay_mult = 0.5f;
            auto max_steps = 1000;
            auto lr_func = get_expon_lr_func(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps);
            BOOST_CHECK_CLOSE(lr_func(0), 0.1f, 1e-5);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_general_utils)
{
    std::cout << "test_general_utils" << std::endl;
    test_build_rotation();
    test_build_scaling_rotation();
    test_strip_lowerdiag();
    test_get_expon_lr_func();
}
#endif // UNIT_TEST
