#include "utils/graphics_utils.h"
#include <cmath>

auto focal2fov(const double &focal_length_x, const uint64_t &height) -> double
{
    return 2.0 * std::atan(height / (2.0 * focal_length_x));
}