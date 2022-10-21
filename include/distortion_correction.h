#ifndef DISTORTION_CORRECTION_H
#define DISTORTION_CORRECTION_H
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

void CorrectionWithPath(string imagePath, string savePath);
void CorrectionWithMat(Mat &imageSrc, Mat &imageDst);
void meanWhiteBalance(Mat &src, Mat &dst);
void imageCorrection(Mat &src, Mat &dst, vector<Point> centers, int count);

#endif
