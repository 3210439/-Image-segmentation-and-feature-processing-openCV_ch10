#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Point2f pt1, pt2;				// 드래그 시작좌표와 종료좌표
Mat image;

void morphing()					// 드래그 거리만큼 영상 왜곡
{
	Mat dst(image.size(), image.type(), Scalar(0));
	int width = image.cols;

	for (float y = 0; y < image.rows; y++) {
		for (float x = 0; x < image.cols; x += 0.1f)	// 가로방향 미세한 변화 표현
		{
			float ratio;
			if (x < pt1.x) ratio = x / pt1.x;
			else		   ratio = (width - x) / (width - pt1.x);

			float dx = ratio * (pt2.x - pt1.x);						// x 좌표의 변화량
			dst.at<uchar>(y, x + dx) = image.at<uchar>(y, x);		// 목적화소값 지정
		}
	}
	dst.copyTo(image);
	imshow("image", image);
}

void onMouse(int event, int x, int y, int flags, void* param)	// 마우스 이벤트 콜백 함수
{
	if (event == EVENT_LBUTTONDOWN) {
		pt1 = Point2f(x, y);				// 드래그 시작 좌표
	}
	else if (event == EVENT_LBUTTONUP) {
		pt2 = Point2f(x, y);				// 드래그 종료 좌표
		morphing();							// 드래그 종료시 워핑 수행
	}
}

int main()
{
	image = imread("./castle.jpg", 0);
	CV_Assert(image.data);

	imshow("image", image);
	setMouseCallback("image", onMouse);			// 마우스 콜백 함수 등록
	waitKey();
	return 0;
}