#include "yolov5_dnn.h"

void YOLOv5Detector::initConfig(std::string onnxpath, int iw, int ih, float threshold) {
    this->input_w = iw;
    this->input_h = ih;
    this->threshold_score = threshold;
    this->net = cv::dnn::readNetFromONNX(onnxpath);
}

void YOLOv5Detector::detect(cv::Mat& frame, std::vector<DetectResult>& results) {
    // 图锟斤拷预锟斤拷锟斤拷 - 锟斤拷式锟斤拷锟斤拷锟斤拷
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    float x_factor = image.cols / 640.0f;
    float y_factor = image.rows / 640.0f;

    // 锟斤拷锟斤拷
    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    cv::Mat preds = this->net.forward();

    // 锟斤拷锟斤拷, 1x25200x85
    // std::cout << "rows: "<< preds.size[1]<< " data: " << preds.size[2] << std::endl;
    cv::Mat det_output(preds.size[1], preds.size[2], CV_32F, preds.ptr<float>());
    float confidence_threshold = 0.5;
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for (int i = 0; i < det_output.rows; i++) {
        float confidence = det_output.at<float>(i, 4);
        if (confidence < 0.45) {
            continue;
        }
        // 锟斤拷锟斤拷训锟斤拷只锟斤拷一锟斤拷锟斤拷锟斤拷锟斤拷锟絚olRange锟斤拷[5,6),前锟斤拷锟侥革拷锟斤拷box锟斤拷锟斤拷
        cv::Mat classes_scores = det_output.row(i).colRange(5, 6);
        cv::Point classIdPoint;
        double score;
        minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

        // 锟斤拷锟脚讹拷 0锟斤拷1之锟斤拷
        if (score > this->threshold_score)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(score);
        }
    }

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
    for (size_t i = 0; i < indexes.size(); i++) {
        DetectResult dr;
        int index = indexes[i];
        int idx = classIds[index];
        dr.box = boxes[index];
        dr.classId = idx;
        dr.score = confidences[index];
        cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
            cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
        results.push_back(dr);
    }

    std::ostringstream ss;
    std::vector<double> layersTimings;
    double freq = cv::getTickFrequency() / 1000.0;
    double time = net.getPerfProfile(layersTimings) / freq;
    ss << "FPS: " << 1000 / time << " ; time : " << time << " ms";
    putText(frame, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 2, 8);
}
