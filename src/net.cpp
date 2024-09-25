#include "torch/csrc/jit/serialization/import.h"
#include <net.h>

Net::Net(void) {
    std::cout << "Net::Net start" << std::endl;
}

Net::~Net(void) {
    std::cout << "Net::Net end" << std::endl;
}

bool Net::chkTensor(at::Tensor tensor)
{
    // 检查输出是否位于GPU上
    if (tensor.is_cuda()) {
        std::cout << "Output is on GPU." << std::endl;
        std::cout << "Output shape: " << tensor.sizes() << std::endl; // 显示形状
        return true;
    } else {
        std::cout << "Output is on CPU." << std::endl;
        return false;
    }
    return true;
}

bool Net::loadNet(){
    std::cout << "Net::loadNet() start" << std::endl;
    try {
        // 禁用梯度计算，因为我们只是加载模型
        torch::NoGradGuard no_grad;

        try {
            Net::module = torch::jit::load("../../model/model_0924.pt");
            std::cout << "Successfully loaded model from " << "model_0924.pt" << std::endl;
        } catch (const c10::Error& e) {
            // PyTorch的C++ API使用c10::Error来报告错误
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    } catch (...) {
        // 捕获其他所有类型的异常
        std::cerr << "An unknown error occurred while loading the model." << std::endl;
        return false;
    }
    std::cout << "Net::loadNet() end" << std::endl;
    return true;
}

bool Net::loadNetFromPath(std::string path){
    std::cout << "Net::loadNetFromPath() start" << std::endl;
    try {
        // 禁用梯度计算，因为我们只是加载模型
        torch::NoGradGuard no_grad;
        try {
            Net::module = torch::jit::load(path);
            std::cout << "Successfully loaded model from " << path << std::endl;
        } catch (const c10::Error& e) {
            // PyTorch的C++ API使用c10::Error来报告错误
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    } catch (...) {
        // 捕获其他所有类型的异常
        std::cerr << "An unknown error occurred while loading the model." << std::endl;
        return false;
    }
    std::cout << "Net::loadNetFromPath() end" << std::endl;
    return true;
}

bool Net::readImage(){
    std::cout << "Net::readImage() start" << std::endl;
    try {
        Net::image397u16 = cv::imread("../../image/image397.tiff", CV_16UC1);
        Net::image398u16 = cv::imread("../../image/image398.tiff", CV_16UC1);
        Net::image399u16 = cv::imread("../../image/image399.tiff", CV_16UC1);

    } catch (...) {
        std::cout << "readImage error." << std::endl;
        return false;
    }
    std::cout << "Net::readImage() end" << std::endl;
    return true;
}

bool Net::readImage(std::string path397, std::string path398, std::string path399){
    std::cout << "Net::readImage() start" << std::endl;
    try {
        Net::image397u16 = cv::imread(path397, CV_16UC1);
        Net::image398u16 = cv::imread(path398, CV_16UC1);
        Net::image399u16 = cv::imread(path399, CV_16UC1);

    } catch (...) {
        std::cout << "readImage error." << std::endl;
        return false;
    }
    std::cout << "Net::readImage() end" << std::endl;
    return true;
}

bool Net::forward(cv::Mat image397u16, cv::Mat image398u16, cv::Mat image399u16) {
    std::cout << "Net::forward start" << std::endl;
    try {
        auto tensor1 = preprocessImage(image397u16);
        auto tensor2 = preprocessImage(image398u16);
        auto tensor3 = preprocessImage(image399u16);

        try{
            Net::module.eval();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(tensor1);
            // inputs.push_back(tensor2);
            // inputs.push_back(tensor3);

            for (const auto& ivalue : inputs) {
                if (ivalue.isTensor()) {
                    const torch::Tensor& tensor = ivalue.toTensor();
                    std::cout << "Inputs shape: " << tensor.sizes() << std::endl;
                }
            }

            at::Tensor output = Net::module.forward(inputs).toTensor();
            std::cout << "Output shape: " << output.sizes() << std::endl; //  [1, 30]

            auto flat_output = output.reshape({-1}); // -1 表示自动计算该维度的大小

            for (int64_t i = 0; i < flat_output.numel(); ++i) {
                std::cout << "Element " << i << ": " << flat_output[i].item<float>() << std::endl;
            }

        }catch (const c10::Error& e) {
            std::cerr << "Caught an exception: " << e.what() << std::endl;
            return false;

        } catch (...) {
            std::cerr << "Unknown exception caught" << std::endl;
            return false;
        }

    } catch (...) {
        std::cerr << "ProcessImage exception caught" << std::endl;
        return false;
    }

    std::cout << "Net::forward end" << std::endl;
    return true;
}


torch::Tensor Net::preprocessImage(const cv::Mat& image) {
    std::cout << "Net::preprocessImage start" << std::endl;
    int startRow = std::max(0, (image.rows - 128) / 2);
    int startCol = std::max(0, (image.cols - 128) / 2);
    cv::Mat croppedImage = image(cv::Rect(startCol, startRow, 128, 128));
    auto tensor = torch::from_blob(croppedImage.data, {1, 1, croppedImage.rows, croppedImage.cols}, torch::kShort);

    tensor = tensor.to(torch::kFloat32).div(65535.0);

    // debug
    // std::cout << "Tensor shape: ";
    // for (int64_t dim : tensor.sizes()) {
    //     std::cout << dim << " ";
    // }
    // std::cout << std::endl;

    auto twoChannelTensor = torch::stack({tensor, tensor}, 1).reshape({1, 2, 128, 128});

    // debug
    // std::cout << "Tensor shape: ";
    // for (int64_t dim : twoChannelTensor.sizes()) {
    //     std::cout << dim << " ";
    // }
    // std::cout << std::endl;

    // return twoChannelTensor.permute({0, 2, 3, 1});
    // return tensor.permute({0, 2, 3, 1});

    std::cout << "Net::preprocessImage end" << std::endl;
    return twoChannelTensor;
}

at::Tensor Net::getZernikeFromImage(cv::Mat image397u16, cv::Mat image398u16, cv::Mat image399u16, int nums) {
    std::cout << "Net::getZernikeFromImage start" << std::endl;

    auto tensor1 = preprocessImage(image397u16);
    auto tensor2 = preprocessImage(image398u16);
    auto tensor3 = preprocessImage(image399u16);

    Net::module.eval();
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor1);
    // inputs.push_back(tensor2);
    // inputs.push_back(tensor3);

    for (const auto& ivalue : inputs) {
        if (ivalue.isTensor()) {
            const torch::Tensor& tensor = ivalue.toTensor();
            std::cout << "Inputs shape: " << tensor.sizes() << std::endl;
        }
    }

    at::Tensor output = Net::module.forward(inputs).toTensor();
    std::cout << "Output shape: " << output.sizes() << std::endl; //  [1, 30]

    auto flat_output = output.reshape({-1}); // -1 表示自动计算该维度的大小

    for (int64_t i = 0; i < flat_output.numel(); ++i) {
        std::cout << "Element " << i << ": " << flat_output[i].item<float>() << std::endl;
    }

    // 创建一个形状为 [1, 65] 的全零张量
    torch::Tensor padded_output = torch::zeros({1, nums}, output.options()); // 保持与 output 相同的 dtype 和 device

    // 将 output 的数据复制到 padded_output 的相应部分
    padded_output.narrow(1, 0, 30).copy_(output);

    // 输出结果
    std::cout << "Padded Output shape: " << padded_output.sizes() << std::endl;

    auto flat_padded_output = padded_output.reshape({-1}); // -1 表示自动计算该维度的大小

    for (int64_t i = 0; i < flat_padded_output.numel(); ++i) {
        std::cout << "Element " << i << ": " << flat_padded_output[i].item<float>() << std::endl;
    }


    std::cout << "Net::getZernikeFromImage end" << std::endl;
    return padded_output;
}


at::Tensor Net::getZernikeFromImage(cv::Mat image397u16, cv::Mat image398u16, cv::Mat image399u16) {
    std::cout << "Net::getZernikeFromImage start" << std::endl;

    auto tensor1 = preprocessImage(image397u16);
    auto tensor2 = preprocessImage(image398u16);
    auto tensor3 = preprocessImage(image399u16);

    Net::module.eval();
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor1);
    // inputs.push_back(tensor2);
    // inputs.push_back(tensor3);

    for (const auto& ivalue : inputs) {
        if (ivalue.isTensor()) {
            const torch::Tensor& tensor = ivalue.toTensor();
            std::cout << "Inputs shape: " << tensor.sizes() << std::endl;
        }
    }

    at::Tensor output = Net::module.forward(inputs).toTensor();
    std::cout << "Output shape: " << output.sizes() << std::endl; //  [1, 30]

    auto flat_output = output.reshape({-1}); // -1 表示自动计算该维度的大小

    for (int64_t i = 0; i < flat_output.numel(); ++i) {
        std::cout << "Element " << i << ": " << flat_output[i].item<float>() << std::endl;
    }

    // 创建一个形状为 [1, 65] 的全零张量
    torch::Tensor padded_output = torch::zeros({1, 65}, output.options()); // 保持与 output 相同的 dtype 和 device

    // 将 output 的数据复制到 padded_output 的相应部分
    padded_output.narrow(1, 0, 30).copy_(output);

    // 输出结果
    std::cout << "Padded Output shape: " << padded_output.sizes() << std::endl;

    auto flat_padded_output = padded_output.reshape({-1}); // -1 表示自动计算该维度的大小

    for (int64_t i = 0; i < flat_padded_output.numel(); ++i) {
        std::cout << "Element " << i << ": " << flat_padded_output[i].item<float>() << std::endl;
    }


    std::cout << "Net::getZernikeFromImage end" << std::endl;
    return padded_output;
}

at::Tensor Net::getZernikeFromImageCuda(cv::Mat image397u16, cv::Mat image398u16, cv::Mat image399u16, int nums) {
    std::cout << "Net::getZernikeFromImageCuda start" << std::endl;

    auto tensor1 = preprocessImage(image397u16).to(at::kCUDA);
    auto tensor2 = preprocessImage(image398u16).to(at::kCUDA);
    auto tensor3 = preprocessImage(image399u16).to(at::kCUDA);

    Net::module.eval();
    Net::module.to(at::kCUDA);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor1);
    // inputs.push_back(tensor2);
    // inputs.push_back(tensor3);

    for (const auto& ivalue : inputs) {
        if (ivalue.isTensor()) {
            const torch::Tensor& tensor = ivalue.toTensor();
            std::cout << "Inputs shape: " << tensor.sizes() << std::endl;
        }
    }

    at::Tensor output = Net::module.forward(inputs).toTensor();
    std::cout << "Output shape: " << output.sizes() << std::endl; //  [1, 30]

    Net::chkTensor(output);

    auto flat_output = output.reshape({-1}); // -1 表示自动计算该维度的大小

    for (int64_t i = 0; i < flat_output.numel(); ++i) {
        std::cout << "Element " << i << ": " << flat_output[i].item<float>() << std::endl;
    }

    // 创建一个形状为 [1, 65] 的全零张量
    torch::Tensor padded_output = torch::zeros({1, nums}, output.options()); // 保持与 output 相同的 dtype 和 device

    // 将 output 的数据复制到 padded_output 的相应部分
    padded_output.narrow(1, 0, 30).copy_(output);

    // 输出结果
    std::cout << "Padded Output shape: " << padded_output.sizes() << std::endl;

    auto flat_padded_output = padded_output.reshape({-1}); // -1 表示自动计算该维度的大小

    for (int64_t i = 0; i < flat_padded_output.numel(); ++i) {
        std::cout << "Element " << i << ": " << flat_padded_output[i].item<float>() << std::endl;
    }

    Net::chkTensor(padded_output);

    std::cout << "Net::getZernikeFromImageCuda end" << std::endl;
    return padded_output;
}


at::Tensor Net::getZernikeFromImageCuda(cv::Mat image397u16, cv::Mat image398u16, cv::Mat image399u16) {
    std::cout << "Net::getZernikeFromImageCuda start" << std::endl;

    auto tensor1 = preprocessImage(image397u16).to(at::kCUDA);
    auto tensor2 = preprocessImage(image398u16).to(at::kCUDA);
    auto tensor3 = preprocessImage(image399u16).to(at::kCUDA);

    Net::chkTensor(tensor1);

    Net::module.eval();
    Net::module.to(at::kCUDA);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor1);
    // inputs.push_back(tensor2);
    // inputs.push_back(tensor3);

    for (const auto& ivalue : inputs) {
        if (ivalue.isTensor()) {
            const torch::Tensor& tensor = ivalue.toTensor();
            std::cout << "Inputs shape: " << tensor.sizes() << std::endl;
        }
    }

    at::Tensor output = Net::module.forward(inputs).toTensor();
    std::cout << "Output shape: " << output.sizes() << std::endl; //  [1, 30]

    Net::chkTensor(output);

    auto flat_output = output.reshape({-1}); // -1 表示自动计算该维度的大小

    for (int64_t i = 0; i < flat_output.numel(); ++i) {
        std::cout << "Element " << i << ": " << flat_output[i].item<float>() << std::endl;
    }

    // 创建一个形状为 [1, 65] 的全零张量
    torch::Tensor padded_output = torch::zeros({1, 65}, output.options()); // 保持与 output 相同的 dtype 和 device

    // 将 output 的数据复制到 padded_output 的相应部分
    padded_output.narrow(1, 0, 30).copy_(output);

    // 输出结果
    std::cout << "Padded Output shape: " << padded_output.sizes() << std::endl;

    auto flat_padded_output = padded_output.reshape({-1}); // -1 表示自动计算该维度的大小

    for (int64_t i = 0; i < flat_padded_output.numel(); ++i) {
        std::cout << "Element " << i << ": " << flat_padded_output[i].item<float>() << std::endl;
    }

    Net::chkTensor(padded_output);

    std::cout << "Net::getZernikeFromImageCuda end" << std::endl;
    return padded_output;
}
