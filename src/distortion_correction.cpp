#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "yolov5_dnn.h"
#include "distortion_correction.h"

#define DETECTION_MODEL_PATH "./models/value.onnx"

void CorrectionWithPath(string imagePath, string savePath) {
	// create the value detection class
	shared_ptr<YOLOv5Detector> detector(new YOLOv5Detector());
	detector->initConfig(DETECTION_MODEL_PATH, 640, 640, 0.25f);
	vector<DetectResult> results;

	// read images
	Mat imageSrc = imread(imagePath);
    Mat imageDetection = imageSrc.clone();
	// image detect
	detector->detect(imageDetection, results);
	// float centers[results.size()][2];
	vector<Point> centers;
	int count = 0;
	for (DetectResult dr : results) {
		Rect box = dr.box;
		putText(imageDetection, to_string(dr.classId), Point(box.tl().x, box.tl().y - 10), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0, 0, 0));
		centers.push_back(Point((box.tl().x + box.br().x) / 2, (box.tl().y + box.br().y) / 2));
		count++;
	}

	Mat imageWhiteBalanceCorrection = imageSrc.clone();
	Mat imageDst = imageSrc.clone();
	// process image with mean white balance
	meanWhiteBalance(imageSrc, imageWhiteBalanceCorrection);
	imageCorrection(imageWhiteBalanceCorrection, imageDst, centers, count);
    imwrite(savePath, imageDst);

	// imshow("test", imageDetection);
	// imshow("test2", imageDst);
	// waitKey(0);

	// return 0;
}

void CorrectionWithMat(Mat &imageSrc, Mat &imageDst) {
	// create the value detection class
	shared_ptr<YOLOv5Detector> detector(new YOLOv5Detector());
	detector->initConfig(DETECTION_MODEL_PATH, 640, 640, 0.25f);
	vector<DetectResult> results;

	// read images
    Mat imageDetection = imageSrc.clone();
	// image detect
	detector->detect(imageDetection, results);
	// float centers[results.size()][2];
	vector<Point> centers;
	int count = 0;
	for (DetectResult dr : results) {
		Rect box = dr.box;
		putText(imageDetection, to_string(dr.classId), Point(box.tl().x, box.tl().y - 10), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0, 0, 0));
		centers.push_back(Point((box.tl().x + box.br().x) / 2, (box.tl().y + box.br().y) / 2));
		count++;
	}

	Mat imageWhiteBalanceCorrection = imageSrc.clone();
	imageDst = imageSrc.clone();
	// process image with mean white balance
	meanWhiteBalance(imageSrc, imageWhiteBalanceCorrection);
	imageCorrection(imageWhiteBalanceCorrection, imageDst, centers, count);

	// imshow("test", imageDetection);
	// imshow("test2", imageDst);
	// waitKey(0);

	// return 0;
}

void meanWhiteBalance(Mat &src, Mat &dst) {
	// split the image into its channels
	vector<Mat> channels;
	split(src, channels);
	float bAvg = mean(channels[0])[0];
	float gAvg = mean(channels[1])[0];
	float rAvg = mean(channels[2])[0];
	float k = (rAvg + gAvg + bAvg) / 3;
	float kr = k/rAvg;
	float kg = k/gAvg;
	float kb = k/bAvg;
	vector<Mat> newChannels(channels.size());
	addWeighted(channels[0], kb, 0, 0, 0, newChannels[0]);
	addWeighted(channels[1], kg, 0, 0, 0, newChannels[1]);
	addWeighted(channels[2], kr, 0, 0, 0, newChannels[2]);
	merge(newChannels, dst);
}

void imageCorrection(Mat &src, Mat &dst, vector<Point> centers, int count) {
	Mat imageMask = Mat::zeros(src.size(), CV_8UC3);
	Mat imageMask1 = Mat::zeros(src.size(), CV_8UC3);
	dst = src.clone();
	// fit ellipse
	// if the number of points is less than 5, the ellipse cannot be fitted
	if (centers.size() < 5) {
		cout << "The number of detected value is less than 3, please check the image." << endl;
		return;
	}
	RotatedRect ellipseResult = fitEllipse(centers);
	// draw ellipse
	ellipse(imageMask, ellipseResult, Scalar(255, 0, 0), 1, 8);
	// draw circle
	circle(imageMask, ellipseResult.center, max(ellipseResult.size.width, ellipseResult.size.height)/2, Scalar(0, 0, 255), 1, 8, 0);
	// get the min x, y & max x, y of the centers
	int minX = src.cols;
	int minY = src.rows;
	int maxX = 0;
	int maxY = 0;
	for (Point p : centers) {
		if (p.x < minX) {
			minX = p.x;
		}
		if (p.x > maxX) {
			maxX = p.x;
		}
		if (p.y < minY) {
			minY = p.y;
		}
		if (p.y > maxY) {
			maxY = p.y;
		}
	}
	float range_x = (maxX - minX) / 2;
	float range_y = (maxY - minY) / 2;
	int rect_y = int(minY - range_y);
	if (rect_y < 0) {
		rect_y = 0;
	}
	int rect_x = int(minX - range_x);
	if (rect_x < 0) {
		rect_x = 0;
	}
	Rect rect = Rect(rect_x, rect_y, int(range_x * 4), int(range_y * 4));
	dst = src(rect);

	// 由于最开始的图像拍摄位置不正，因此通过下述代码进行图像畸变校正，现拍摄位置改为正位，因此不再进行畸变校正
	// 仪表图像稍微旋转不会影响仪表的刻度提取，起始、终止刻度的提取根据极坐标变换后刻度间的距离进行判断
	/* 
	float k1 = tan((180 - ellipseResult.angle) * CV_PI / 180);
	float k2 = tan((ellipseResult.angle - 90) * CV_PI / 180);
	float x_0 = ellipseResult.center.x, y_0 = ellipseResult.center.y;
	float x_11 = int(-y_0 / k1 + x_0);
	float x_12 = int((src.rows - y_0) / k1 + x_0);
	float y_11 = int(k2 * (0 - x_0) + y_0);
	float y_12 = int(k2 * (src.cols - x_0) + y_0);

	// draw line 1
	line(imageMask1, Point(x_11, 0), Point(x_12, src.rows), Scalar(0, 255, 0), 1, 8, 0);
	imageMask += imageMask1;
	// imshow("test1", imageMask);
	// imshow("test2", imageMask1);
	// waitKey(0);

	vector<Point> intersectionEllipse;
	vector<Point> intersectionCircle;

	for(int i = 0; i < src.size().height; i++) {
		for(int j = 0; j < src.size().width; j++) {
			if (imageMask.at<Vec3b>(i, j)[0] == 255) {
				intersectionEllipse.push_back(Point(j, i));
			}
			if (imageMask.at<Vec3b>(i, j)[0] == 255 && (imageMask.at<Vec3b>(i, j)[1] == 255)) {
				intersectionEllipse.push_back(Point(j, i));
			}
			if (imageMask.at<Vec3b>(i, j)[2] == 255 && (imageMask.at<Vec3b>(i, j)[1] == 255)) {
				intersectionCircle.push_back(Point(j, i));
			}
		}
	}
	
	float moveVect = min_element(intersectionEllipse.begin(), intersectionEllipse.end(), [](Point a, Point b) {return a.y < b.y; })->y 
					- min_element(intersectionCircle.begin(), intersectionCircle.end(), [](Point a, Point b) {return a.y < b.y; })->y;
	for (int i = 0; i < intersectionCircle.size(); i++) {
		intersectionCircle[i].x += moveVect;
	}

	float buffer = 2 * moveVect;
	Point2f intersectionImagePoints[4] = { Point2f(0, 0), 
											Point2f(src.cols, 0), 
											Point2f(0, src.rows), 
											Point2f(src.cols, src.rows) };
	Point2f intersectionDestPoints[4] = { Point2f(0+buffer, 0), 
											Point2f(src.size().width-buffer, 0),
											Point2f(0-buffer, src.size().height+2*buffer), 
											Point2f(src.size().width+buffer, src.size().height+2*buffer) };
	Mat transformMat = getPerspectiveTransform(intersectionImagePoints, intersectionDestPoints);
	warpPerspective(src, dst, transformMat, Size(src.size().width+buffer, src.size().height+2*buffer));

	// calculate the intersection between line, circle and ellipse
	*/
}