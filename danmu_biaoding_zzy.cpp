#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
using namespace cv;
using namespace std;

int main()
{

	//����ͼƬ��ַ���ı�·��
	string infilename = "qipan/filename.txt";
	//������Hд����ı�·��
	string outfilename = "qipan/caliberation_result.txt";

	//����ifstream���󣬶����Ӳ�̶����ڴ�
	ifstream fin(infilename);
	//����ofstream���� �����ڴ�д��Ӳ��
	ofstream fout(outfilename);


	//ͼ������
	int imageCount = 0;
	//ͼ��ߴ�
	cv::Size imageSize;
	//�궨����ÿ��ÿ�еĽǵ���
	cv::Size boardSize = cv::Size(8, 6);

	//����ÿ��ͼ���ϼ�⵽�Ľǵ�
	std::vector<Point2f>  imagePointsBuf;
	//�����⵽�����нǵ�
	std::vector<std::vector<Point2f>> imagePointsSeq;

	//�����ļ�����
	std::vector<std::string>  filenames;
	char filename[100];
	std::cout << "��ʼ��ȡ�ǵ�......" << std::endl;
	if (fin.is_open())
	{
		while (!fin.eof())
		{
			//һ�ζ�ȡһ��
			fin.getline(filename, sizeof(filename) / sizeof(char));
			//�����ļ���
			filenames.push_back(filename);
			//��ȡͼƬ
			Mat imageInput = cv::imread(filename);
			//�����һ��ͼƬʱ��ȡͼ�����Ϣ
			if (imageCount == 0)
			{
				imageSize.width = imageInput.cols;
				imageSize.height = imageInput.rows;
				std::cout << "imageSize.width = " << imageSize.width << std::endl;
				std::cout << "imageSize.height = " << imageSize.height << std::endl;
			}

			//�ۼ�ͼƬ����
			imageCount++;

			//��ȡÿһ��ͼƬ�Ľǵ�
			if (cv::findChessboardCorners(imageInput, boardSize, imagePointsBuf) == 0)
			{
				//�Ҳ����ǵ�
				std::cout << "Can not find chessboard corners!" << std::endl;
				exit(1);
			}
			else
			{
				Mat viewGray;
				//ת��Ϊ�Ҷ�ͼƬ
				cv::cvtColor(imageInput, viewGray, cv::COLOR_BGR2GRAY);
				//�����ؾ�ȷ��   �Դ���ȡ�Ľǵ���о�ȷ��
				cv::find4QuadCornerSubpix(viewGray, imagePointsBuf, cv::Size(5, 5));
				//���������ص�
				imagePointsSeq.push_back(imagePointsBuf);
				//��ͼ������ʾ�ǵ�λ��
				//cv::drawChessboardCorners(viewGray, boardSize, imagePointsBuf, true);
				//��ʾͼƬ
				//cv::imshow("Camera Calibration", viewGray);
				//cv::imwrite("test.jpg", viewGray);
				//�ȴ�0.5s
				//waitKey(500);
			}
		}
		std::cout << "ͼƬ������ = " << imageCount << std::endl;
		//����ÿ��ͼƬ�ϵĽǵ��� 48
		int cornerNum = boardSize.width * boardSize.height;

		//�ǵ�����
		int total = imagePointsSeq.size()*cornerNum;

		std::cout << "��ʼ�궨" << std::endl;



		//����ͼƬ�ǵ�����
		std::vector<int> pointCounts;
		//����궨���Ͻǵ����ά����
		std::vector<std::vector<cv::Point3f>> objectPoints;
		//������ڲ������� M=[fx �� u0,0 fy v0,0 0 1]
		cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64F, Scalar::all(0));
		//�������5������ϵ��k1,k2,p1,p2,k3
		cv::Mat distCoeffs = cv::Mat(1, 5, CV_64F, Scalar::all(0));
		//ÿ��ͼƬ����ת����
		std::vector<cv::Mat> tvecsMat;
		//ÿ��ͼƬ��ƽ������
		std::vector<cv::Mat> rvecsMat;

		//��ʼ���궨���Ͻǵ����ά����
		int i, j, t;
		for (t = 0; t < imageCount; t++)
		{
			std::vector<cv::Point3f> tempPointSet;
			//����
			for (i = 0; i < boardSize.height; i++)
			{
				//����
				for (j = 0; j < boardSize.width; j++)
				{
					cv::Point3f realPoint;
					//����궨�������������ϵ��z=0��ƽ���ϡ�
					realPoint.x = i*24.5;
					realPoint.y = j*24.5;
					realPoint.z = 0;
					tempPointSet.push_back(realPoint);
				}
			}
			objectPoints.push_back(tempPointSet);
		}
		//cout << "��ά������:" << std::endl;
		//cout << objectPoints[0] << std::endl << std::endl;

		//��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨��
		for (i = 0; i < imageCount; i++)
		{
			pointCounts.push_back(boardSize.width*boardSize.height);
		}
		//��ʼ�궨
		cv::calibrateCamera(objectPoints, imagePointsSeq, imageSize, cameraMatrix, distCoeffs, rvecsMat, tvecsMat);
		std::cout << "�궨���" << std::endl;

		//����ÿ��ͼ�����ת����
		cv::Mat rotationMatrix = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));
		cout << "����ڲ�������:" << std::endl;
		cout << cameraMatrix << std::endl << std::endl;
		cout << "����ϵ��:" << std::endl;
		cout << distCoeffs << std::endl << std::endl;

		//�����һ��ͼƬ�Ĳ�������һ��ͼƬ����ȽϹ���
		cout << "�����һ��ͼƬ�Ĳ�������һ��ͼƬ����ȽϹ���" << endl;
		cout << "��" << 1 << "��ͼ�����ת����:" << std::endl;
		cout << rvecsMat[0] << std::endl;
		//����ת����ת��Ϊ���Ӧ����ת����
		cv::Rodrigues(rvecsMat[0], rotationMatrix);
		cout << "��" << 1 << "��ͼ�����ת����:" << std::endl;
		cout << rotationMatrix << std::endl;
		cout << "��" << 1 << "��ͼ���ƽ������:" << std::endl;
		cout << tvecsMat[0] << std::endl;

		//��ΪZwΪ0������R3����Ҫ��ֻ����R1��R2��ֵ
		cv::Mat R2 = cv::Mat(3, 2, CV_64FC1, Scalar::all(0));
		R2.at<double>(0, 0) = rotationMatrix.at<double>(0, 0);
		R2.at<double>(0, 1) = rotationMatrix.at<double>(0, 1);
		R2.at<double>(1, 0) = rotationMatrix.at<double>(1, 0);
		R2.at<double>(1, 1) = rotationMatrix.at<double>(1, 1);
		R2.at<double>(2, 0) = rotationMatrix.at<double>(2, 0);
		R2.at<double>(2, 1) = rotationMatrix.at<double>(2, 1);
		//cout << "R2" << std::endl;
		//cout << R2 << std::endl;

		//�õ���ξ���R1��R2��T
		cv::Mat R2T = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));
		hconcat(R2, tvecsMat[0], R2T);
		//cout << "R2T" << std::endl;
		//cout << R2T << std::endl;

		//�õ���ά�ռ����꣨Xw,Yw��ת������ά��������ľ���M
		Mat M;
		M = cameraMatrix* R2T;
		cout << "��" << 1 << "��ͼ�����ά�ռ����꣨Xw,Yw��ת������ά��������ľ���M:" << std::endl;
		cout << M << std::endl;

		//�õ���ά��������ת������ά�ռ����꣨Xw,Yw���ľ���H
		Mat H;
		H = M.inv();
		cout << "��" << 1 << "��ά��������ת������ά�ռ����꣨Xw,Yw���ľ���H:" << std::endl;
		cout << H << std::endl;

		//����ά�������괢�浽������
		cv::Mat point_2d = cv::Mat(3, 48, CV_64F, Scalar::all(0));
		for (size_t i = 0; i < 48; i++)
		{
			point_2d.at<double>(0, i) = imagePointsSeq[0][i].x;
			point_2d.at<double>(1, i) = imagePointsSeq[0][i].y;
			point_2d.at<double>(2, i) = 1;
		}

		//����ά�ռ����괢�浽������
		cv::Mat point_3d = cv::Mat(3, 48, CV_64F, Scalar::all(0));
		for (size_t i = 0; i < 48; i++)
		{
			point_3d.at<double>(0, i) = objectPoints[0][i].x;
			point_3d.at<double>(1, i) = objectPoints[0][i].y;
			point_3d.at<double>(2, i) = 1;
		}
		//����洢ת�����ά���غͿռ�����ľ���
		cv::Mat point_3d_2d = cv::Mat(3, 48, CV_64F, Scalar::all(0));
		cv::Mat point_2d_3d = cv::Mat(3, 48, CV_64F, Scalar::all(0));

		//���㾭��M����ת����õ��Ķ�ά��������
		point_3d_2d = M*point_3d;
		for (size_t i = 0; i < 48; i++)
		{
			point_3d_2d.at<double>(0, i) = point_3d_2d.at<double>(0, i) / point_3d_2d.at<double>(2, i);
			point_3d_2d.at<double>(1, i) = point_3d_2d.at<double>(1, i) / point_3d_2d.at<double>(2, i);
			point_3d_2d.at<double>(2, i) = point_3d_2d.at<double>(2, i) / point_3d_2d.at<double>(2, i);
		}
		//cout << "��ԭ��2d����" << std::endl;
		//cout << point_3d_2d << std::endl;

		//���㾭��H����ת����õ�����ά�ռ�����
		point_2d_3d = H*point_2d;
		for (size_t i = 0; i < 48; i++)
		{
			point_2d_3d.at<double>(0, i) = point_2d_3d.at<double>(0, i) / point_2d_3d.at<double>(2, i);
			point_2d_3d.at<double>(1, i) = point_2d_3d.at<double>(1, i) / point_2d_3d.at<double>(2, i);
			point_2d_3d.at<double>(2, i) = point_2d_3d.at<double>(2, i) / point_2d_3d.at<double>(2, i);
		}
		//cout << "��ԭ��3d����" << std::endl;
		//cout << point_2d_3d << std::endl;

		//���۵�һ��ͼ�����
		cout << "���۵�һ��ͼ�����" << endl;
		cv::Mat x_wucha = cv::Mat(48, 1, CV_64F, Scalar::all(0));
		cv::Mat y_wucha = cv::Mat(48, 1, CV_64F, Scalar::all(0));

		//����
		double x_wucha_total = 0;
		double y_wucha_total = 0;

		//ƽ�����
		double x_wucha_pinjun = 0;
		double y_wucha_pinjun = 0;

		//������
		double x_wucha_max = 0;
		double y_wucha_max = 0;
		double* x_wucha_max_p = &x_wucha_max;
		double* y_wucha_max_p = &y_wucha_max;

		//��С���
		double x_wucha_min = 0;
		double y_wucha_min = 0;
		double* x_wucha_min_p = &x_wucha_min;
		double* y_wucha_min_p = &y_wucha_min;

		//��������
		for (size_t i = 0; i < 48; i++)
		{
			x_wucha.at<double>(i, 0) = fabs(point_3d.at<double>(0, i) - point_2d_3d.at<double>(0, i));
			x_wucha_total = x_wucha_total + x_wucha.at<double>(i, 0);
			y_wucha.at<double>(i, 0) = fabs(point_3d.at<double>(1, i) - point_2d_3d.at<double>(1, i));
			y_wucha_total = y_wucha_total + y_wucha.at<double>(i, 0);

		}
		//����ƽ�����
		x_wucha_pinjun = x_wucha_total / 48.0;
		y_wucha_pinjun = y_wucha_total / 48.0;

		//���������С���
		minMaxIdx(x_wucha, x_wucha_min_p, x_wucha_max_p);
		minMaxIdx(y_wucha, y_wucha_min_p, y_wucha_max_p);

		cout << "����x��y������ֵ��" << std::endl;
		cout << x_wucha_max << std::endl;
		cout << y_wucha_max << std::endl;

		cout << "���������Сֵ��" << std::endl;
		cout << x_wucha_min << std::endl;
		cout << y_wucha_min << std::endl;

		cout << "����x��y���ƽ��ֵ��" << std::endl;
		cout << x_wucha_pinjun << std::endl;
		cout << y_wucha_pinjun << std::endl;

		//�������ն�ά����������ռ�����ת���ľ���H
		fout << H << endl;
		//�ͷ���Դ
		fin.close();
		fout.close();
		system("pause");
	}
}
