#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include "distortion_correction.h"
#include "sf6_meter_reader.h"

#define SEGMENTATION_MODEL_PATH "./models/segmentation.onnx"
#define MIN_METER_SCALE -0.1
#define MAX_METER_SCALE 0.9


void imageSegmentation(cv::Mat& imageSrc, cv::Mat& dst) {
	Mat image = imageSrc.clone();
	resize(image, image, Size(512, 512));
	cvtColor(image, image, COLOR_BGR2RGB);
	// image = image/127.5 -1
	image.convertTo(image, CV_64FC3, 1.0 / 127.5, -1);

	dnn::Net net = dnn::readNetFromONNX(SEGMENTATION_MODEL_PATH);
	image = image.reshape(1, {1, 512, 512, 3});
	net.setInput(image);
	Mat output = net.forward();
	// cout << output.size << endl;
	output = output.reshape(1, {512, 512, 3});
	dst = Mat(output.size[0], output.size[1], CV_8UC1);
	// argmax(output, axis=2)
	for (int i = 0; i < output.size[0]; i++) {
		for (int j = 0; j < output.size[1]; j++) {
			float* data = output.ptr<float>(i, j);
			int maxIndex = 0;
			float maxValue = data[0];
			for (int k = 1; k < output.size[2]; k++) {
				// cout << data[k] << endl;
				if (data[k] > maxValue) {
					maxValue = data[k];
					maxIndex = k;
				}
			}
			// if you want to get the mask, the ellements should be 0 or 255
			// dst.at<uchar>(i, j) = maxIndex * 127.5;
			dst.at<uchar>(i, j) = maxIndex;
		}
	}
	// imshow("dst", dst);
	// waitKey(0);
}

void imageDenoising(cv::Mat& imageSrc, cv::Mat& dst) {
	medianBlur(imageSrc, dst, 5);
}

void polarCoordinate(Point& pointSrc, Point& pointDst, Point2f& pointCenter, float Kangle, float Klin) {
	/*
	transform the point from cartesian coordinate to polar coordinate
	:param pointSrc: the point in cartesian coordinate
	:param pointDst: the point in polar coordinate
	:param pointCenter: the center of the polar coordinate
	:param Kangle: the coefficient of angle
	:param Klin: the coefficient of linear
	*/
	float x0 = pointSrc.x - pointCenter.x;
	float y0 = pointSrc.y - pointCenter.y;
	// attention: the y axis is from top to bottom
	float angle = atan2(y0, x0);
	if (x0>0 && y0>0) {
		angle = angle;
	}
	else if (x0<0 && y0>0) {
		angle = angle;
	}else if (x0<0 && y0<0) {
		angle = 2*CV_PI + angle;
	}else if (x0>0 && y0<0) {
		angle = 2*CV_PI + angle;
	}
	float phi = Kangle * angle;
	float rho = Klin * sqrt((pointSrc.x - pointCenter.x) * (pointSrc.x - pointCenter.x)
	 			+ (pointSrc.y - pointCenter.y) * (pointSrc.y - pointCenter.y));
	pointDst = Point(rho, phi);
}

void groupPoints(vector<Point>& points, int* groupLable, int* groupIndex) {
	/*
	group the points into different groups
	:param points: the points to be grouped
	:param groupLable: the lable of each point
	:param groupIndex: the index of the first zero class
	*/
	// calculate the distance between each adjacent points
	vector<float> distance;
	for (int i = 0; i < points.size() - 1; i++) {
		distance.push_back(sqrt((points[i].x - points[i + 1].x) * (points[i].x - points[i + 1].x)
			+ (points[i].y - points[i + 1].y) * (points[i].y - points[i + 1].y)));
	}
	float maxDistanceIndex = max_element(distance.begin(), distance.end()) - distance.begin();
	maxDistanceIndex += 1;
	float maxDistance = *max_element(distance.begin(), distance.end());
	float minDistance = *min_element(distance.begin(), distance.end());
	if (maxDistance > 5 * minDistance) {
		// set groupLable[:maxDistanceIndex] as class 0
		for (int i = 0; i < maxDistanceIndex; i++) {
			*(groupLable + i) = 0;
		}
		// set groupLable[maxDistanceIndex:] as class 1
		for (int i = maxDistanceIndex; i < points.size(); i++) {
			*(groupLable + i) = 1;
		}
	} else {
		// set groupLable[:] as class 0
		for (int i = 0; i < points.size(); i++) {
			*(groupLable + i) = 0;
		}
	}
	*groupIndex = maxDistanceIndex;
}

void meterReading(cv::Mat& image, cv::Mat& mask, cv::Mat& result, float& value) {
	Mat predictionMask = mask.clone();
	// get the point elements of the mask
	// set 2 = 0 in predictionMask
	for (int i = 0; i < predictionMask.rows; i++) {
		for (int j = 0; j < predictionMask.cols; j++) {
			if (predictionMask.at<uchar>(i, j) == 2) {
				predictionMask.at<uchar>(i, j) = 0;
			}
		}
	}
	Mat predictionMaskPoint;
	imageDenoising(predictionMask, predictionMaskPoint);
	resize(predictionMaskPoint, predictionMaskPoint, image.size(), 0, 0, INTER_NEAREST);
	imwrite("./temp/predictionMaskPoint.png", predictionMaskPoint);

	// get the scale elements of the mask
	predictionMask = mask.clone();
	// set 1 = 0 in predictionMask
	for (int i = 0; i < predictionMask.rows; i++) {
		for (int j = 0; j < predictionMask.cols; j++) {
			if (predictionMask.at<uchar>(i, j) == 1) {
				predictionMask.at<uchar>(i, j) = 0;
			}
		}
	}
	Mat predictionMaskScale;
	imageDenoising(predictionMask, predictionMaskScale);
	resize(predictionMaskScale, predictionMaskScale, image.size(), 0, 0, INTER_NEAREST);
	imwrite("./temp/predictionMaskScale.png", predictionMaskScale);

	// predictionMaskScale processing
	predictionMaskScale /= 2;
	// set predictionMaskScale to CV_8UC1
	predictionMaskScale.convertTo(predictionMaskScale, CV_8UC1);
	// get all the contours of the scale
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(predictionMaskScale, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	// get the minAreaRect of the scale
	vector<RotatedRect> minRect(contours.size());
	vector<Point> centers(contours.size());
	
	for (int i = 0; i < contours.size(); i++) {
		minRect[i] = minAreaRect(contours[i]);
		Mat boxPts;
		vector<Point> boxPtsPoint;
		boxPoints(minRect[i], boxPts);
		// conver boxPts to Point
		for (int j = 0; j < boxPts.rows; j++) {
			boxPtsPoint.push_back(Point(boxPts.at<float>(j, 0), boxPts.at<float>(j, 1)));
		}
		// fill all minRect
		// fillConvexPoly(image, boxPtsPoint, Scalar(245, 255, 0), LINE_8, 0);
		centers[i] = minRect[i].center;
	}
	if (centers.size() < 5) {
		cout << "The scale is not detected!" << endl;
		return;
	}
	// fitEllipse with centers
	RotatedRect ellipseParams = fitEllipse(centers);
	// filter the point far from the ellipse center, fit ellipse again
	vector<Point> centersFilter;
	for (int i = 0; i < centers.size(); i++) {
		if (sqrt((centers[i].x - ellipseParams.center.x) * (centers[i].x - ellipseParams.center.x)
			+ (centers[i].y - ellipseParams.center.y) * (centers[i].y - ellipseParams.center.y)) < 1.2 * ellipseParams.size.width) {
			centersFilter.push_back(centers[i]);
		}
	}
	if (centersFilter.size() < 5) {
		cout << "The scale is not detected!" << endl;
		return;
	}
	ellipseParams = fitEllipse(centersFilter);
	// ellipse(image, ellipseParams, Scalar(255, 0, 0), 2);

	// predictionMaskPoint processing
	findContours(predictionMaskPoint, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	if (contours.size() == 0) {
		return;
	}
	vector<Point> pointHeart;
	vector<RotatedRect> minRectPoint(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		minRectPoint[i] = minAreaRect(contours[i]);
		// caculate the area of the minRectPoint
		float area = minRectPoint[i].size.width * minRectPoint[i].size.height;
		// if the area is too small, it is not the point
		if (area < 60) {
			continue;
		}
		Mat boxPts;
		vector<Point> boxPtsPoint;
		boxPoints(minRectPoint[i], boxPts);
		// conver boxPts to Point
		for (int j = 0; j < boxPts.rows; j++) {
			boxPtsPoint.push_back(Point(boxPts.at<float>(j, 0), boxPts.at<float>(j, 1)));
		}
		// fill all minRectPoint
		// fillConvexPoly(image, boxPtsPoint, Scalar(48, 48, 255), LINE_8, 0);
		pointHeart.push_back(minRectPoint[i].center);
	}
	// if the pointHeart is not unique, means the model detect not only one point, can not process, return
	if (pointHeart.size() > 3 || pointHeart.size() < 1) {
		cout << pointHeart.size() << endl;
		return;
	}
	// warpPolar with image
	Mat imagePolar;
	warpPolar(image, imagePolar, Size(-1,-1), ellipseParams.center, ellipseParams.size.height, WARP_POLAR_LINEAR + INTER_LINEAR);
	vector<Point> valueCentersPolar;
	// caculate Kangle and Klin
	float Kangle = imagePolar.rows / (2 * CV_PI);
	float Klin = imagePolar.cols / (ellipseParams.size.height);
	for (int i = 0; i < centersFilter.size(); i++) {
		Point pointDst;
		polarCoordinate(centersFilter[i], pointDst, ellipseParams.center, Kangle, Klin);
		valueCentersPolar.push_back(pointDst);
		// draw circle for each valueCentersPolar
		circle(imagePolar, pointDst, 3, Scalar(254, 255, 0), -1);
	}
	// get the polar coordinate of the point_heart
	Point pointHeartPolar;
	polarCoordinate(pointHeart[0], pointHeartPolar, ellipseParams.center, Kangle, Klin);
	// draw circle for pointHeartPolar
	circle(imagePolar, pointHeartPolar, 3, Scalar(254, 0, 255), -1);
	// sort the valueCentersPolar by y
	sort(valueCentersPolar.begin(), valueCentersPolar.end(), [](Point a, Point b) {return a.y < b.y; });

	int index = 0;
	int valueCentersPolarLabel[valueCentersPolar.size()];
	groupPoints(valueCentersPolar, valueCentersPolarLabel, &index);

	for (int i = 0; i < valueCentersPolar.size(); i++) {
		if (valueCentersPolarLabel[i] == 0) {
			circle(imagePolar, valueCentersPolar[i], 3, Scalar(0, 0, 255), -1);
		} else {
			circle(imagePolar, valueCentersPolar[i], 3, Scalar(0, 255, 0), -1);
		}
	}
	// the crop coordinate of y is the y of the contours_centers_polar[label_index]
    // cut the image_pular by the crop coordinate
	Mat imagePolarFormer = imagePolar(Rect(0, valueCentersPolar[index].y, imagePolar.cols, imagePolar.rows - valueCentersPolar[index].y));
	Mat imagePolarLatter = imagePolar(Rect(0, 0, imagePolar.cols, valueCentersPolar[index].y));
	// concatenate the image_pular_former and image_pular_latter, exchange the former and latter
	Mat imagePolarExchange(imagePolar.rows, imagePolar.cols, CV_8UC3);
	vconcat(imagePolarFormer, imagePolarLatter, imagePolarExchange);
	// convert point_heart_polar
	if (pointHeartPolar.y > valueCentersPolar[index].y) {
		pointHeartPolar.y -= valueCentersPolar[index].y;
	} else {
		pointHeartPolar.y += imagePolarFormer.rows;
	}

	// put the valueCentersPolarExchange before the index to the end of the vector
	vector<Point> valueCentersPolarExchange;

	for (int i = 0; i < valueCentersPolar.size(); i++) {
		if (i < index) {
			valueCentersPolarExchange.push_back(valueCentersPolar[i]);
			valueCentersPolarExchange[i].y += imagePolarFormer.rows;
		} else if (i >= index) {
			valueCentersPolarExchange.push_back(valueCentersPolar[i]);
			valueCentersPolarExchange[i].y -= valueCentersPolar[index].y;
		}
	}
	sort(valueCentersPolarExchange.begin(), valueCentersPolarExchange.end(), [](Point a, Point b) {return a.y < b.y; });
	

	// highlight the start and end point of the contours_centers_polar
	circle(imagePolarExchange, valueCentersPolarExchange[0], 4, Scalar(0, 255, 255), -1);
	circle(imagePolarExchange, valueCentersPolarExchange[valueCentersPolarExchange.size() - 1], 4, Scalar(0, 255, 255), -1);
	// highlight the point_heart_polar
	circle(imagePolarExchange, pointHeartPolar, 3, Scalar(254, 0, 255), -1);

	// calculate the reading of the heart rate
	float readRate = 1.0 * pointHeartPolar.y / (valueCentersPolarExchange[valueCentersPolarExchange.size() - 1].y - valueCentersPolarExchange[0].y);
	float meterValue = readRate * (MAX_METER_SCALE - MIN_METER_SCALE) + MIN_METER_SCALE;
	if (meterValue > MAX_METER_SCALE) {
		meterValue = MAX_METER_SCALE;
	} else if (meterValue < MIN_METER_SCALE) {
		meterValue = MIN_METER_SCALE;
	}
	putText(imagePolarExchange, to_string(meterValue), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
	value = meterValue;
	result = imagePolarExchange;
	
	/*
	imwrite("./temp/imagePolarExchange.png", imagePolarExchange);
	imshow("image", image);
	imshow("imagePolar", imagePolar);
	imshow("imagePolarExchange", imagePolarExchange);
	waitKey(0);
	*/
}
