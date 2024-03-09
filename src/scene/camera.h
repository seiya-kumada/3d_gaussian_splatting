#include <torch/torch.h>

class Camera : public torch::nn::Module
{
private:
    int uid_;
    int colmap_id_;
};