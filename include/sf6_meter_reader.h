#ifndef SF6_METER_READER_H
#define SF6_METER_READER_H
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

void imageSegmentation(cv::Mat& image, cv::Mat& mask);
void imageDenoising(cv::Mat& imageSrc, cv::Mat& dst);
void meterReading(cv::Mat& image, cv::Mat& mask, cv::Mat& result, float& value);
void polarCoordinate(Point& pointSrc, Point& pointDst, Point& pointCenter, float Kangle, float Klin);

#endif
