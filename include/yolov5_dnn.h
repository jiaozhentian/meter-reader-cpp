#ifndef YOLOR5_DNN_H
#define YOLOR5_DNN_H

#pragma once

#include <opencv2/opencv.hpp>
struct DetectResult {
    int classId;
    float score;
    cv::Rect box;
};

class YOLOv5Detector {
public:
    void initConfig(std::string onnxpath, int iw, int ih, float threshold);
    void detect(cv::Mat& frame, std::vector<DetectResult>& result);
private:
    int input_w = 640;
    int input_h = 640;
    cv::dnn::Net net;
    int threshold_score = 0.25;
};

#endif // YOLOR5_DNN_H