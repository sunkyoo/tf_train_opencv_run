//
// Author: Sunkyoo Hwang / sunkyoo.hwang at gmail.com
// Handwritten digit recognition using OpenCV dnn module
//

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

void on_mouse(int event, int x, int y, int flags, void* userdata);

int main()
{
	Net net = readNet("../mnist_cnn.pb");

	if (net.empty()) {
		cerr << "Network load failed!" << endl;
		return -1;
	}

	Mat img = Mat::zeros(400, 400, CV_8UC1);

	imshow("img", img);
	setMouseCallback("img", on_mouse, (void*)&img);
	
	cout << "Draw a digit with your mouse and press SPACEBAR." << endl;
	cout << "Press \'c\' to erase all, and ESC to quit." << endl;

	while (true) {
		int c = waitKey(0);

		if (c == 27) {
			break;
		} else if (c == ' ') {
			// MNIST : 28x28 floating-point images, [0, 1].
			Mat blob = blobFromImage(img, 1/255.f, Size(28, 28));
			net.setInput(blob);
			Mat prob = net.forward();

			vector<double> layersTimings;
			double inf_ms = net.getPerfProfile(layersTimings) * 1000 / getTickFrequency();

			double maxVal;
			Point maxLoc;
			minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc);
			int digit = maxLoc.x;

			cout << digit << " (" << maxVal * 100 << "%) (" << inf_ms << "ms.)" <<endl;

			img.setTo(0);
			imshow("img", img);
		} else if (c == 'c') {
			img.setTo(0);
			imshow("img", img);
		}
	}

	return 0;
}

Point ptPrev(-1, -1);

void on_mouse(int event, int x, int y, int flags, void* userdata)
{
	Mat img = *(Mat*)userdata;

	if (event == EVENT_LBUTTONDOWN) {
		ptPrev = Point(x, y);
	} else if (event == EVENT_LBUTTONUP) {
		ptPrev = Point(-1, -1);
	} else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
		line(img, ptPrev, Point(x, y), Scalar::all(255), 40, LINE_AA, 0);
		ptPrev = Point(x, y);

		imshow("img", img);
	}
}