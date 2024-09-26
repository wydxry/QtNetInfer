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

// 模拟FFT shift的函数
void fftshift(cv::Mat& input) {
    int rows = input.rows;
    int cols = input.cols;
    int centerRow = rows / 2;
    int centerCol = cols / 2;

    // 交换四个象限
    cv::Mat q0(input, cv::Rect(0, 0, centerCol, centerRow));
    cv::Mat q1(input, cv::Rect(centerCol, 0, cols - centerCol, centerRow));
    cv::Mat q2(input, cv::Rect(0, centerRow, centerCol, rows - centerRow));
    cv::Mat q3(input, cv::Rect(centerCol, centerRow, cols - centerCol, rows - centerRow));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}


bool Net::getFeatureFromImage() {
    std::cout << "Net::getFeatureFromImage start" << std::endl;
    try{
        // 读取图像
        cv::Mat image1 = cv::imread("../../image/Fimg1_up.png", cv::IMREAD_GRAYSCALE);
        cv::Mat image2 = cv::imread("../../image/Fimg1_sub.png", cv::IMREAD_GRAYSCALE);

        if (image1.empty() || image2.empty()) {
            qDebug() << "Could not open or find the image";
            return false;
        }

        try{
            // 对第一张图像进行傅里叶变换
            cv::Mat planes1[] = {cv::Mat_<float>(image1), cv::Mat::zeros(image1.size(), CV_32F)};
            cv::Mat complexI1;
            cv::merge(planes1, 2, complexI1);
            cv::dft(complexI1, complexI1);

            // 对第二张图像进行傅里叶变换
            cv::Mat planes2[] = {cv::Mat_<float>(image2), cv::Mat::zeros(image2.size(), CV_32F)};
            cv::Mat complexI2;
            cv::merge(planes2, 2, complexI2);
            cv::dft(complexI2, complexI2);

            // 应用FFT shift
            fftshift(complexI1);
            fftshift(complexI2);

            try{
                // 分离实部和虚部
                std::vector<cv::Mat> tmp1, tmp2;
                cv::split(complexI1, tmp1);
                cv::split(complexI2, tmp2);

                // 获取实部和虚部
                cv::Mat realI1 = tmp1[0];
                cv::Mat imagI1 = tmp1[1];
                cv::Mat realI2 = tmp2[0];
                cv::Mat imagI2 = tmp2[1];

                // 点除并取模（注意：这里需要处理除以零的情况）
                cv::Mat resultReal, resultImag;
                cv::divide(realI1, realI2, resultReal, 1.0, CV_32F); // 添加小的epsilon值以避免除以零
                cv::divide(imagI1, imagI2, resultImag, 1.0, CV_32F);

                // 计算复数结果的模长
                cv::Mat resultMagnitude;
                cv::magnitude(resultReal, resultImag, resultMagnitude);

                // 归一化并保存结果图像
                cv::normalize(resultMagnitude, resultMagnitude, 0, 255, cv::NORM_MINMAX, CV_8U);
                cv::imwrite("../../image/feature1.jpg", resultMagnitude);
            } catch(...) {
                std::cout << "feature error" << std::endl;
            }

            // 分离第一张图像的幅度和相位
            cv::split(complexI1, planes1);
            cv::Mat magnitude1, phase1;
            cv::magnitude(planes1[0], planes1[1], magnitude1);
            cv::cartToPolar(planes1[0], planes1[1], magnitude1, phase1, true);

            // 分离第二张图像的幅度和相位
            cv::split(complexI2, planes2);
            cv::Mat magnitude2, phase2;
            cv::magnitude(planes2[0], planes2[1], magnitude2);
            cv::cartToPolar(planes2[0], planes2[1], magnitude2, phase2, true);

            // 对幅度谱进行对数变换和归一化以改善可视化效果（可选）
            cv::Mat logMagnitude1, logMagnitude2;
            magnitude1 += cv::Scalar::all(1);
            magnitude2 += cv::Scalar::all(1);
            cv::log(magnitude1, logMagnitude1);
            cv::log(magnitude2, logMagnitude2);
            cv::normalize(logMagnitude1, logMagnitude1, 0, 255, cv::NORM_MINMAX);
            cv::normalize(logMagnitude2, logMagnitude2, 0, 255, cv::NORM_MINMAX);

            // 转换为8位无符号整数并保存
            logMagnitude1.convertTo(logMagnitude1, CV_8U);
            logMagnitude2.convertTo(logMagnitude2, CV_8U);

            // 保存图像
            cv::imwrite("../../image/magnitude_image1.jpg", magnitude1);
            cv::imwrite("../../image/phase_image1.png", phase1);
            cv::imwrite("../../image/magnitude_image2.jpg", magnitude2);
            cv::imwrite("../../image/phase_image2.png", phase2);
            cv::imwrite("../../image/log_magnitude_image1.jpg", logMagnitude1);
            cv::imwrite("../../image/log_magnitude_image2.jpg", logMagnitude2);

            // 缩放相位谱到0-255范围
            phase1 = (phase1 + CV_PI) / (2 * CV_PI) * 255;
            phase2 = (phase2 + CV_PI) / (2 * CV_PI) * 255;
            phase1.convertTo(phase1, CV_8U);
            phase2.convertTo(phase2, CV_8U);

            cv::imwrite("../../image/phase_image1_convert.png", phase1);
            cv::imwrite("../../image/phase_image2_convert.png", phase2);

        } catch(...) {
            std::cout << "process error" << std::endl;
        }

    } catch(...) {
        std::cout << "read error" << std::endl;
        return false;
    }

    std::cout << "Net::getFeatureFromImage end" << std::endl;
    return true;
}


bool Net::testInferTime() {
    // Net::loadNet();
    Net::loadNetFromPath("../../model/model_0927_50_PE.pt");

    int epoch = 1000;

    module.to(torch::kCUDA); // 默认设置为GPU

    // 创建一个随机的输入张量模拟数据
    torch::Tensor input = torch::randn({1, 2, 128, 128}).to(torch::kCPU);

    // 将张量转换为 IValue 并放入 vector 中
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // CPU测速
    module.to(torch::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epoch; ++i) {
        torch::Tensor output = module.forward(inputs).toTensor();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CPU infer time: " << elapsed_seconds.count() / epoch << " s" << std::endl;
    std::cout << "CPU infer time: " << duration_cpu * 1.0 / epoch << " ms" << std::endl;
    std::cout << "CPU fps: " << epoch / elapsed_seconds.count() << std::endl;

    // GPU测速
    module.to(torch::kCUDA);
    input = input.to(torch::kCUDA);
    inputs.clear();
    inputs.push_back(input);
    // 预热
    for (int i = 0; i < 10; ++i) {
        torch::Tensor output = module.forward(inputs).toTensor();
    }
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epoch; ++i) {
        torch::Tensor output = module.forward(inputs).toTensor();
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "GPU infer time: " << elapsed_seconds.count() / epoch << " s" << std::endl;
    std::cout << "GPU infer time: " << duration_gpu * 1.0 / epoch  << " ms" << std::endl;
    std::cout << "GPU fps: " << epoch / elapsed_seconds.count() << std::endl;

    return true;
}
