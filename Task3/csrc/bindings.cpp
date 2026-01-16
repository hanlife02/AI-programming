#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <tuple>

namespace py = pybind11;

torch::Tensor conv2d_forward(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b, int64_t stride,
                             int64_t padding);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv2d_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor w,
    bool need_grad_x,
    bool need_grad_w,
    bool need_grad_b,
    int64_t stride,
    int64_t padding);

torch::Tensor relu_forward(torch::Tensor x);
torch::Tensor relu_backward(torch::Tensor grad_out, torch::Tensor x);

std::tuple<torch::Tensor, torch::Tensor> maxpool2d_forward(torch::Tensor x, int64_t kernel, int64_t stride);
torch::Tensor maxpool2d_backward(torch::Tensor grad_out, torch::Tensor indices, int64_t in_h, int64_t in_w,
                                 int64_t kernel, int64_t stride);

torch::Tensor global_avg_pool2d_forward(torch::Tensor x);
torch::Tensor global_avg_pool2d_backward(torch::Tensor grad_out, int64_t h, int64_t w);

torch::Tensor linear_forward(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor w,
    bool need_grad_x,
    bool need_grad_w,
    bool need_grad_b);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> batchnorm2d_forward(torch::Tensor x, torch::Tensor weight,
                                                                            torch::Tensor bias, torch::Tensor running_mean,
                                                                            torch::Tensor running_var, bool training,
                                                                            double momentum, double eps);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> batchnorm2d_backward(torch::Tensor grad_out, torch::Tensor x,
                                                                             torch::Tensor weight, torch::Tensor saved_mean,
                                                                             torch::Tensor saved_invstd, bool need_grad_x,
                                                                             bool need_grad_w, bool need_grad_b);

std::tuple<torch::Tensor, torch::Tensor> cross_entropy_forward(torch::Tensor logits, torch::Tensor targets);
torch::Tensor cross_entropy_backward(torch::Tensor probs, torch::Tensor targets, torch::Tensor grad_out);

void sgd_update_(torch::Tensor param, torch::Tensor grad, c10::optional<torch::Tensor> velocity, double lr,
                 double momentum, double weight_decay);

struct Conv2dOp {
    int64_t stride;
    int64_t padding;
    bool has_bias;

    Conv2dOp(int64_t stride_, int64_t padding_, bool has_bias_) : stride(stride_), padding(padding_), has_bias(has_bias_) {}

    torch::Tensor forward(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b) const {
        return conv2d_forward(x, w, b, stride, padding);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(torch::Tensor grad_out, torch::Tensor x,
                                                                      torch::Tensor w, bool need_grad_x,
                                                                      bool need_grad_w, bool need_grad_b) const {
        return conv2d_backward(grad_out, x, w, need_grad_x, need_grad_w, need_grad_b, stride, padding);
    }
};

struct BatchNorm2dOp {
    double momentum;
    double eps;

    BatchNorm2dOp(double momentum_, double eps_) : momentum(momentum_), eps(eps_) {}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                                                    torch::Tensor running_mean, torch::Tensor running_var,
                                                                    bool training) const {
        return batchnorm2d_forward(x, weight, bias, running_mean, running_var, training, momentum, eps);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(torch::Tensor grad_out, torch::Tensor x,
                                                                     torch::Tensor weight, torch::Tensor saved_mean,
                                                                     torch::Tensor saved_invstd, bool need_grad_x,
                                                                     bool need_grad_w, bool need_grad_b) const {
        return batchnorm2d_backward(grad_out, x, weight, saved_mean, saved_invstd, need_grad_x, need_grad_w, need_grad_b);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &conv2d_forward, "conv2d forward (CUDA)");
    m.def("conv2d_backward", &conv2d_backward, "conv2d backward (CUDA)");
    m.def("relu_forward", &relu_forward, "relu forward (CUDA)");
    m.def("relu_backward", &relu_backward, "relu backward (CUDA)");
    m.def("maxpool2d_forward", &maxpool2d_forward, "maxpool2d forward (CUDA)");
    m.def("maxpool2d_backward", &maxpool2d_backward, "maxpool2d backward (CUDA)");
    m.def("global_avg_pool2d_forward", &global_avg_pool2d_forward, "global avg pool2d forward (CUDA)");
    m.def("global_avg_pool2d_backward", &global_avg_pool2d_backward, "global avg pool2d backward (CUDA)");
    m.def("linear_forward", &linear_forward, "linear forward (CUDA)");
    m.def("linear_backward", &linear_backward, "linear backward (CUDA)");
    m.def("batchnorm2d_forward", &batchnorm2d_forward, "batchnorm2d forward (CUDA)");
    m.def("batchnorm2d_backward", &batchnorm2d_backward, "batchnorm2d backward (CUDA)");
    m.def("cross_entropy_forward", &cross_entropy_forward, "cross entropy forward (CUDA)");
    m.def("cross_entropy_backward", &cross_entropy_backward, "cross entropy backward (CUDA)");
    m.def("sgd_update_", &sgd_update_, "SGD update in-place (CUDA)");

    py::class_<Conv2dOp>(m, "Conv2dOp")
        .def(py::init<int64_t, int64_t, bool>(), py::arg("stride"), py::arg("padding"), py::arg("has_bias"))
        .def("forward", &Conv2dOp::forward, py::arg("x"), py::arg("w"), py::arg("b") = py::none())
        .def("backward", &Conv2dOp::backward, py::arg("grad_out"), py::arg("x"), py::arg("w"),
             py::arg("need_grad_x"), py::arg("need_grad_w"), py::arg("need_grad_b"));

    py::class_<BatchNorm2dOp>(m, "BatchNorm2dOp")
        .def(py::init<double, double>(), py::arg("momentum") = 0.1, py::arg("eps") = 1e-5)
        .def("forward", &BatchNorm2dOp::forward, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("running_mean"),
             py::arg("running_var"), py::arg("training"))
        .def("backward", &BatchNorm2dOp::backward, py::arg("grad_out"), py::arg("x"), py::arg("weight"),
             py::arg("saved_mean"), py::arg("saved_invstd"), py::arg("need_grad_x"), py::arg("need_grad_w"),
             py::arg("need_grad_b"));
}
