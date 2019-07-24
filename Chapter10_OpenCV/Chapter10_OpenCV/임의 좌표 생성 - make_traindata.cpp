#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void make_trainData(Mat trainData, Mat group[2], Mat& classLable);

void draw_points(Mat& image, Mat group[2]);

int main()
{
	int Nsample = 100;
	Mat trainData(Nsample, 2, CV_32FC1, Scalar(0));			// 학습데이터
	Mat classLable(Nsample, 1, CV_32FC1, Scalar(0));		// 레이블 값 행렬
	Mat image(400, 400, CV_8UC3, Scalar(255, 255, 255));

	Mat group[2];									
	make_trainData(trainData, group, classLable);	// 학습데이터 랜덤 저장
	draw_points(image, group);						// 학습데이터 표시
	imshow("학습데이터", image);
	waitKey();
	return 0;
}

void draw_points(Mat& image, Mat group[2]) {
	for (int i = 0; i < group[0].rows; i++)
	{
		Point2f pt1(group[0].at<float>(i, 0), group[0].at<float>(i, 1));	// 윗부분 절반
		Point2f pt2(group[1].at<float>(i, 0), group[1].at<float>(i, 1));	
		circle(image, pt1, 3, Scalar(0, 0, 255), FILLED);					// 빨간색 원
		circle(image, pt2, 3, Scalar(0, 255, 0), FILLED);					// 녹색 원
	}
}

void make_trainData(Mat trainData, Mat group[2], Mat& classLable)
{
	int half = trainData.rows / 2;
	Range r1(0, half);							// 윗 부분 절반 범위
	Range r2(half, trainData.rows);				// 아랫부분 절반 범위
	group[0] = trainData.rowRange(r1);			// 입력행렬의 위 절반
	group[1] = trainData.rowRange(r2);			// 입력행렬의 아래 절반

	randn(group[0], 150, 50);					// 임의값 지정 - 평균 200, 표준편차 50
	randn(group[1], 250, 50);
	classLable.rowRange(r1).setTo(0);			// 레이블 행렬 위 절반에 0 지정
	classLable.rowRange(r2).setTo(1);			// 레이블 행렬 아래 절반에 1 지정
}