#ifndef NET_H
#define NET_H

#undef slots
#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>
#define slots Q_SLOTS

#include <opencv.hpp>
#include <QString>
#include <QDebug>
#include <iostream>

class Net
{

private:
    torch::jit::script::Module module;

public:
    cv::Mat image397u16;
    cv::Mat image398u16;
    cv::Mat image399u16;

    Net();
    ~Net();

    bool loadNet();
    bool loadNetFromPath(std::string path);

    bool testInferTime();

    bool readImage();
    bool readImage(std::string path397, std::string path398, std::string path399);

    bool getFeatureFromImage();

    torch::Tensor preprocessImage(const cv::Mat& image);

    bool forward(cv::Mat image397u16, cv::Mat image398u16, cv::Mat image399u16);

    bool chkTensor(at::Tensor tensor);

    at::Tensor getZernikeFromImage(cv::Mat image397u16, cv::Mat image398u16, cv::Mat image399u16);
    at::Tensor getZernikeFromImage(cv::Mat image397u16, cv::Mat image398u16, cv::Mat image399u16, int nums);

    at::Tensor getZernikeFromImageCuda(cv::Mat image397u16, cv::Mat image398u16, cv::Mat image399u16);
    at::Tensor getZernikeFromImageCuda(cv::Mat image397u16, cv::Mat image398u16, cv::Mat image399u16, int nums);
};

#endif // NET_H
