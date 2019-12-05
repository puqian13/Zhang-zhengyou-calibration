#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
using namespace cv;
using namespace std;

int main()
{

	//保存图片地址的文本路径
	string infilename = "qipan/filename.txt";
	//最后矩阵H写入的文本路径
	string outfilename = "qipan/caliberation_result.txt";

	//创建ifstream对象，对象从硬盘读到内存
	ifstream fin(infilename);
	//创建ofstream对象 ，从内存写到硬盘
	ofstream fout(outfilename);


	//图像数量
	int imageCount = 0;
	//图像尺寸
	cv::Size imageSize;
	//标定板上每行每列的角点数
	cv::Size boardSize = cv::Size(8, 6);

	//缓存每幅图像上检测到的角点
	std::vector<Point2f>  imagePointsBuf;
	//保存检测到的所有角点
	std::vector<std::vector<Point2f>> imagePointsSeq;

	//保存文件名称
	std::vector<std::string>  filenames;
	char filename[100];
	std::cout << "开始提取角点......" << std::endl;
	if (fin.is_open())
	{
		while (!fin.eof())
		{
			//一次读取一行
			fin.getline(filename, sizeof(filename) / sizeof(char));
			//保存文件名
			filenames.push_back(filename);
			//读取图片
			Mat imageInput = cv::imread(filename);
			//读入第一张图片时获取图宽高信息
			if (imageCount == 0)
			{
				imageSize.width = imageInput.cols;
				imageSize.height = imageInput.rows;
				std::cout << "imageSize.width = " << imageSize.width << std::endl;
				std::cout << "imageSize.height = " << imageSize.height << std::endl;
			}

			//累加图片数量
			imageCount++;

			//提取每一张图片的角点
			if (cv::findChessboardCorners(imageInput, boardSize, imagePointsBuf) == 0)
			{
				//找不到角点
				std::cout << "Can not find chessboard corners!" << std::endl;
				exit(1);
			}
			else
			{
				Mat viewGray;
				//转换为灰度图片
				cv::cvtColor(imageInput, viewGray, cv::COLOR_BGR2GRAY);
				//亚像素精确化   对粗提取的角点进行精确化
				cv::find4QuadCornerSubpix(viewGray, imagePointsBuf, cv::Size(5, 5));
				//保存亚像素点
				imagePointsSeq.push_back(imagePointsBuf);
				//在图像上显示角点位置
				//cv::drawChessboardCorners(viewGray, boardSize, imagePointsBuf, true);
				//显示图片
				//cv::imshow("Camera Calibration", viewGray);
				//cv::imwrite("test.jpg", viewGray);
				//等待0.5s
				//waitKey(500);
			}
		}
		std::cout << "图片总数量 = " << imageCount << std::endl;
		//计算每张图片上的角点数 48
		int cornerNum = boardSize.width * boardSize.height;

		//角点总数
		int total = imagePointsSeq.size()*cornerNum;

		std::cout << "开始标定" << std::endl;



		//毎幅图片角点数量
		std::vector<int> pointCounts;
		//保存标定板上角点的三维坐标
		std::vector<std::vector<cv::Point3f>> objectPoints;
		//摄像机内参数矩阵 M=[fx γ u0,0 fy v0,0 0 1]
		cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64F, Scalar::all(0));
		//摄像机的5个畸变系数k1,k2,p1,p2,k3
		cv::Mat distCoeffs = cv::Mat(1, 5, CV_64F, Scalar::all(0));
		//每幅图片的旋转向量
		std::vector<cv::Mat> tvecsMat;
		//每幅图片的平移向量
		std::vector<cv::Mat> rvecsMat;

		//初始化标定板上角点的三维坐标
		int i, j, t;
		for (t = 0; t < imageCount; t++)
		{
			std::vector<cv::Point3f> tempPointSet;
			//行数
			for (i = 0; i < boardSize.height; i++)
			{
				//列数
				for (j = 0; j < boardSize.width; j++)
				{
					cv::Point3f realPoint;
					//假设标定板放在世界坐标系中z=0的平面上。
					realPoint.x = i*24.5;
					realPoint.y = j*24.5;
					realPoint.z = 0;
					tempPointSet.push_back(realPoint);
				}
			}
			objectPoints.push_back(tempPointSet);
		}
		//cout << "三维点坐标:" << std::endl;
		//cout << objectPoints[0] << std::endl << std::endl;

		//初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板
		for (i = 0; i < imageCount; i++)
		{
			pointCounts.push_back(boardSize.width*boardSize.height);
		}
		//开始标定
		cv::calibrateCamera(objectPoints, imagePointsSeq, imageSize, cameraMatrix, distCoeffs, rvecsMat, tvecsMat);
		std::cout << "标定完成" << std::endl;

		//保存每张图像的旋转矩阵
		cv::Mat rotationMatrix = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));
		cout << "相机内参数矩阵:" << std::endl;
		cout << cameraMatrix << std::endl << std::endl;
		cout << "畸变系数:" << std::endl;
		cout << distCoeffs << std::endl << std::endl;

		//输出第一幅图片的参数，第一幅图片坐标比较规则
		cout << "输出第一幅图片的参数，第一幅图片坐标比较规则" << endl;
		cout << "第" << 1 << "幅图像的旋转向量:" << std::endl;
		cout << rvecsMat[0] << std::endl;
		//将旋转向量转换为相对应的旋转矩阵
		cv::Rodrigues(rvecsMat[0], rotationMatrix);
		cout << "第" << 1 << "幅图像的旋转矩阵:" << std::endl;
		cout << rotationMatrix << std::endl;
		cout << "第" << 1 << "幅图像的平移向量:" << std::endl;
		cout << tvecsMat[0] << std::endl;

		//因为Zw为0，所以R3不需要，只保留R1和R2的值
		cv::Mat R2 = cv::Mat(3, 2, CV_64FC1, Scalar::all(0));
		R2.at<double>(0, 0) = rotationMatrix.at<double>(0, 0);
		R2.at<double>(0, 1) = rotationMatrix.at<double>(0, 1);
		R2.at<double>(1, 0) = rotationMatrix.at<double>(1, 0);
		R2.at<double>(1, 1) = rotationMatrix.at<double>(1, 1);
		R2.at<double>(2, 0) = rotationMatrix.at<double>(2, 0);
		R2.at<double>(2, 1) = rotationMatrix.at<double>(2, 1);
		//cout << "R2" << std::endl;
		//cout << R2 << std::endl;

		//得到外参矩阵R1、R2、T
		cv::Mat R2T = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));
		hconcat(R2, tvecsMat[0], R2T);
		//cout << "R2T" << std::endl;
		//cout << R2T << std::endl;

		//得到三维空间坐标（Xw,Yw）转换到二维像素坐标的矩阵M
		Mat M;
		M = cameraMatrix* R2T;
		cout << "第" << 1 << "幅图像的三维空间坐标（Xw,Yw）转换到二维像素坐标的矩阵M:" << std::endl;
		cout << M << std::endl;

		//得到二维像素坐标转换到三维空间坐标（Xw,Yw）的矩阵H
		Mat H;
		H = M.inv();
		cout << "第" << 1 << "二维像素坐标转换到三维空间坐标（Xw,Yw）的矩阵H:" << std::endl;
		cout << H << std::endl;

		//将二维像素坐标储存到矩阵中
		cv::Mat point_2d = cv::Mat(3, 48, CV_64F, Scalar::all(0));
		for (size_t i = 0; i < 48; i++)
		{
			point_2d.at<double>(0, i) = imagePointsSeq[0][i].x;
			point_2d.at<double>(1, i) = imagePointsSeq[0][i].y;
			point_2d.at<double>(2, i) = 1;
		}

		//将三维空间坐标储存到矩阵中
		cv::Mat point_3d = cv::Mat(3, 48, CV_64F, Scalar::all(0));
		for (size_t i = 0; i < 48; i++)
		{
			point_3d.at<double>(0, i) = objectPoints[0][i].x;
			point_3d.at<double>(1, i) = objectPoints[0][i].y;
			point_3d.at<double>(2, i) = 1;
		}
		//定义存储转换后二维像素和空间坐标的矩阵
		cv::Mat point_3d_2d = cv::Mat(3, 48, CV_64F, Scalar::all(0));
		cv::Mat point_2d_3d = cv::Mat(3, 48, CV_64F, Scalar::all(0));

		//计算经过M矩阵转换后得到的二维像素坐标
		point_3d_2d = M*point_3d;
		for (size_t i = 0; i < 48; i++)
		{
			point_3d_2d.at<double>(0, i) = point_3d_2d.at<double>(0, i) / point_3d_2d.at<double>(2, i);
			point_3d_2d.at<double>(1, i) = point_3d_2d.at<double>(1, i) / point_3d_2d.at<double>(2, i);
			point_3d_2d.at<double>(2, i) = point_3d_2d.at<double>(2, i) / point_3d_2d.at<double>(2, i);
		}
		//cout << "还原的2d坐标" << std::endl;
		//cout << point_3d_2d << std::endl;

		//计算经过H矩阵转换后得到的三维空间坐标
		point_2d_3d = H*point_2d;
		for (size_t i = 0; i < 48; i++)
		{
			point_2d_3d.at<double>(0, i) = point_2d_3d.at<double>(0, i) / point_2d_3d.at<double>(2, i);
			point_2d_3d.at<double>(1, i) = point_2d_3d.at<double>(1, i) / point_2d_3d.at<double>(2, i);
			point_2d_3d.at<double>(2, i) = point_2d_3d.at<double>(2, i) / point_2d_3d.at<double>(2, i);
		}
		//cout << "还原的3d坐标" << std::endl;
		//cout << point_2d_3d << std::endl;

		//评价第一幅图的误差
		cout << "评价第一幅图的误差" << endl;
		cv::Mat x_wucha = cv::Mat(48, 1, CV_64F, Scalar::all(0));
		cv::Mat y_wucha = cv::Mat(48, 1, CV_64F, Scalar::all(0));

		//误差和
		double x_wucha_total = 0;
		double y_wucha_total = 0;

		//平均误差
		double x_wucha_pinjun = 0;
		double y_wucha_pinjun = 0;

		//最大误差
		double x_wucha_max = 0;
		double y_wucha_max = 0;
		double* x_wucha_max_p = &x_wucha_max;
		double* y_wucha_max_p = &y_wucha_max;

		//最小误差
		double x_wucha_min = 0;
		double y_wucha_min = 0;
		double* x_wucha_min_p = &x_wucha_min;
		double* y_wucha_min_p = &y_wucha_min;

		//计算误差和
		for (size_t i = 0; i < 48; i++)
		{
			x_wucha.at<double>(i, 0) = fabs(point_3d.at<double>(0, i) - point_2d_3d.at<double>(0, i));
			x_wucha_total = x_wucha_total + x_wucha.at<double>(i, 0);
			y_wucha.at<double>(i, 0) = fabs(point_3d.at<double>(1, i) - point_2d_3d.at<double>(1, i));
			y_wucha_total = y_wucha_total + y_wucha.at<double>(i, 0);

		}
		//计算平均误差
		x_wucha_pinjun = x_wucha_total / 48.0;
		y_wucha_pinjun = y_wucha_total / 48.0;

		//计算最大最小误差
		minMaxIdx(x_wucha, x_wucha_min_p, x_wucha_max_p);
		minMaxIdx(y_wucha, y_wucha_min_p, y_wucha_max_p);

		cout << "坐标x，y误差最大值：" << std::endl;
		cout << x_wucha_max << std::endl;
		cout << y_wucha_max << std::endl;

		cout << "坐标误差最小值：" << std::endl;
		cout << x_wucha_min << std::endl;
		cout << y_wucha_min << std::endl;

		cout << "坐标x，y误差平均值：" << std::endl;
		cout << x_wucha_pinjun << std::endl;
		cout << y_wucha_pinjun << std::endl;

		//保存最终二维像素坐标向空间坐标转换的矩阵H
		fout << H << endl;
		//释放资源
		fin.close();
		fout.close();
		system("pause");
	}
}
