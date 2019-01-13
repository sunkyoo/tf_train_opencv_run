#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

Mat img;
Point ptPrev(-1, -1);

void on_mouse(int event, int x, int y, int flags, void*)
{
	if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
		return;
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		ptPrev = Point(-1, -1);
	else if (event == EVENT_LBUTTONDOWN)
		ptPrev = Point(x, y);
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		Point pt(x, y);
		if (ptPrev.x < 0)
			ptPrev = pt;
		line(img, ptPrev, pt, Scalar::all(255), 40, LINE_AA, 0);
		ptPrev = pt;

		imshow("img", img);
	}
}

int main(int argc, char* argv[])
{
	Net net = dnn::readNet("..\\mnist_cnn.pb");

	if (net.empty()) {
		cerr << "Network load failed!" << endl;
		return -1;
	}

	img = Mat::zeros(400, 400, CV_8U);

	imshow("img", img);
	setMouseCallback("img", on_mouse, 0);
	
	cout << "Draw a digit with your mouse and press SPACEBAR." << endl;
	cout << "Press \'c\' to erase all, and ESC to quit." << endl;

	while (1) {
		int c = waitKey(0);

		if (c == 27) {
			break;
		}
		else if (c == ' ') {
			// MNIST : 28x28 floating-point images, [0, 1].
			Mat inputBlob = blobFromImage(img, 1/255.f, Size(28, 28), Scalar(), false);
			net.setInput(inputBlob);
			Mat prob = net.forward();

			vector<double> layersTimings;
			double inf_ms = net.getPerfProfile(layersTimings) * 1000 / getTickFrequency();

			double maxVal;
			Point maxLoc;
			minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc);
			int digit = maxLoc.x;

			cout << digit << " (" << maxVal * 100 << "%) (" << inf_ms << "ms.)" <<endl;

			img = Mat::zeros(400, 400, CV_8U);
			imshow("img", img);
		}
		else if (c == 'c') {
			img = Mat::zeros(400, 400, CV_8U);
			imshow("img", img);
		}
	}

	return 0;
}
