#include <torch/torch.h>

class CameraInfo : public torch::nn::Module
{
private:
    int uid_;
    int colmap_id_;
};