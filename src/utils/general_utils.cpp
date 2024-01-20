#include "src/utils/general_utils.h"

namespace
{
    /**
     * Builds a rotation matrix from the given quaternion.
     *
     * @param r The quaternion representing the rotation.
     * @return The rotation matrix.
     */
    auto build_rotation(const torch::Tensor &r) -> torch::Tensor
    {
        auto norm = torch::sqrt(r.select(1, 0) * r.select(1, 0) +
                                r.select(1, 1) * r.select(1, 1) +
                                r.select(1, 2) * r.select(1, 2) +
                                r.select(1, 3) * r.select(1, 3));

        auto q = r / norm.unsqueeze(1);

        auto device = torch::kCUDA;
        torch::Tensor R = torch::zeros({q.size(0), 3, 3}, torch::TensorOptions().device(device));

        auto rr = q.select(1, 0);
        auto x = q.select(1, 1);
        auto y = q.select(1, 2);
        auto z = q.select(1, 3);

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
}

/**
 * @brief Builds a scaling and rotation matrix.
 *
 * This function takes two tensors, `s` and `r`, and returns a tensor representing the scaling and rotation matrix.
 *
 * @param s The scaling tensor.
 * @param r The rotation tensor.
 * @return The scaling and rotation matrix tensor.
 */
auto build_scaling_rotation(const torch::Tensor &s, const torch::Tensor &r) -> torch::Tensor
{
    auto device = torch::kCUDA;
    torch::Tensor L = torch::zeros({s.size(0), 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device));
    torch::Tensor R = build_rotation(r); // build_rotation関数は先ほど定義したものを使用

    L.index_put_({torch::indexing::Slice(), 0, 0}, s.select(1, 0));
    L.index_put_({torch::indexing::Slice(), 1, 1}, s.select(1, 1));
    L.index_put_({torch::indexing::Slice(), 2, 2}, s.select(1, 2));

    L = torch::matmul(R, L);
    return L;
}

#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(test_general_utils)
{
    BOOST_CHECK_EQUAL(1, 1);
}
#endif // UNIT_TEST
