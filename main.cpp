#include <iostream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include "distortion_correction.h"
#include "sf6_meter_reader.h"

/*
int main(){
	// image segmentation test
	string imagePath = "F:\\My_app\\meter-reader-cpp\\meter-reader\\data\\data_test\\19700101084932.jpg";
	// string savePath = "./temp/test_segmentation.jpg";
	Mat imageSrc = imread(imagePath);
	Mat imageCorrection = imageSrc.clone();
	Mat imageDst = imageSrc.clone();
	Mat imageResult = imageSrc.clone();
	float value = 0.0f;
	// distortion correction
	CorrectionWithMat(imageSrc, imageCorrection);
	// image segmentation
	imageSegmentation(imageCorrection, imageDst);
	// sf6 meter reader
	meterReading(imageCorrection, imageDst, imageResult, value);
	cout << "The value of meter is: " << value << endl;

	imshow("imageSrc", imageSrc);
	imshow("imageDst", imageDst);
	imshow("imageResult", imageResult);
	waitKey(0);
	
	return 0;
}
*/

void GetFileNames(string path, vector<string>& filenames)
{
	DIR *pDir;
	struct dirent *ptr;
	if (!(pDir = opendir(path.c_str()))){
		cout << "Open dir error..." << endl;
		exit(1);
	}
	while((ptr = readdir(pDir)) != 0){
		if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
			filenames.push_back(path + "\\" + ptr->d_name);
		}
	}
	closedir(pDir);
}

int main(){
	vector<string> filenames;
	string path = "./data/data_test";
	GetFileNames(path, filenames);
	for (int i = 0; i < filenames.size(); i++){
		// image segmentation test
		string imagePath = filenames[i];
		// string savePath = "./temp/test_segmentation.jpg";
		Mat imageSrc = imread(imagePath);
		Mat imageCorrection = imageSrc.clone();
		Mat imageDst = imageSrc.clone();
		Mat imageResult = imageSrc.clone();
		float value = 0.0f;
		// distortion correction
		CorrectionWithMat(imageSrc, imageCorrection);
		// imshow("imageCorrection", imageCorrection);
		// waitKey(0);
		// image segmentation
		imageSegmentation(imageCorrection, imageDst);
		// sf6 meter reader
		meterReading(imageCorrection, imageDst, imageResult, value);
		cout << "The value of meter is: " << value << endl;

		// imshow("imageSrc", imageSrc);
		// imshow("imageDst", imageDst);
		// imshow("imageResult", imageResult);
		// waitKey(0);
	}
	
	return 0;
}