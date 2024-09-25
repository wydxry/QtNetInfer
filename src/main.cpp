#include <QCoreApplication>
#include <net.h>
#include <direct.h>

bool test_cur_path() {
    std::cout << "test_cur_path() start" << std::endl;
    char buff[_MAX_PATH];

    _getcwd(buff, _MAX_PATH);

    std::string currPath(buff);

    std::cout << "current path: " << currPath << std::endl;


    std::cout << "test_cur_path() end" << std::endl;
}

bool test_opencv()
{
    std::cout << "test_opencv() start" << std::endl;
    try{
        cv::Mat mat = cv::imread("../../image/1.png");
        if(mat.empty()){
            return false;
        }
        std::cout << mat.channels() << std::endl;
        std::cout << mat.depth() << std::endl;
        cv::imshow("Image", mat);
    }catch(...)
    {
        qWarning() << "cv::imread error." << Qt::endl;
    }

    std::cout << "test_opencv() end" << std::endl;
    return true;
}

bool test_libtorch()
{
    std::cout << "test_libtorch() start" << std::endl;
    try {
        torch::Tensor tensor = torch::rand({ 5,3 });
        std::cout << tensor << std::endl;
        std::cout << torch::cuda::is_available() << std::endl; // 1
    }catch(...)
    {
        qWarning() << "torch error." << Qt::endl;
        return false;
    }
    std::cout << "test_libtorch() end" << std::endl;
    return true;
}

bool test_load_model()
{
    std::cout << "test_load_model() start" << std::endl;
    try{
        torch::NoGradGuard no_grad;
        torch::jit::script::Module module;
        module = torch::jit::load("../../model/model_0924.pt");
        std::cout << "Succeed in loading model" << std::endl;
        try{
            auto input = torch::ones({1, 2, 128, 128}, torch::kFloat32);
            module.eval();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            at::Tensor output = module.forward(inputs).toTensor();
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

    }catch(...) {
        std::cout << "loading model error" << std::endl;
        return false;
    }

    std::cout << "test_load_model() end" << std::endl;
    return true;
}

bool test_cv_readimage() {
    std::cout << "test_cv_readimage() start" << std::endl;
    try{
        cv::Mat image397u8 = cv::imread("../../image/image397.tiff", CV_16UC1);
        cv::Mat image398u8 = cv::imread("../../image/image398.tiff", CV_16UC1);
        cv::Mat image399u8 = cv::imread("../../image/image399.tiff", CV_16UC1);

        std::cout << "channels: " << image397u8.channels() << " dims: " << image397u8.dims << std::endl;
    } catch(...) {
        std::cout << " " << std::endl;
        return false;
    }
    std::cout << "test_cv_readimage() end" << std::endl;
    return true;
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    std::cout << "main start" << std::endl;

    // 测试必要库
    test_cur_path(); // test cur path
    test_opencv(); // test opencv
    test_libtorch(); // test libtorch
    test_load_model(); // test load model
    test_cv_readimage(); // test read image

    std::string modelPath = "../../model/model_0924.pt";

    std::string path397 = "../../image/image397.tiff";
    std::string path398 = "../../image/image398.tiff";
    std::string path399 = "../../image/image399.tiff";

    // 核心
    Net net;
    net.loadNet(); // 默认路径
    net.loadNetFromPath(modelPath); // 自定义路径
    net.readImage();
    net.readImage(path397, path398, path399);
    net.forward(net.image397u16, net.image398u16, net.image399u16);
    net.getZernikeFromImage(net.image397u16, net.image398u16, net.image399u16);  // 65阶
    net.getZernikeFromImage(net.image397u16, net.image398u16, net.image399u16, 75); // 自定义阶数

    net.getZernikeFromImageCuda(net.image397u16, net.image398u16, net.image399u16);  // 65阶
    net.getZernikeFromImageCuda(net.image397u16, net.image398u16, net.image399u16, 75); // 自定义阶数

    std::cout << "main end" << std::endl;
    return a.exec();
}
