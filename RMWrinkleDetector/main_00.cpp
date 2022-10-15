//
//  main.cpp
//  RMWrinkleDetector
//
//  Created by meicet on 2022/10/14.
//

//---海森矩阵二维图像增强
// https://blog.csdn.net/u013921430/article/details/79770458

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
#include <algorithm>    // std::min_element, std::max_element

 
#define STEP 6
#define ABS(X) ((X)>0? X:(-(X)))
#define PI 3.1415926
 
using namespace std;
using namespace cv;
 

int main(int argc, const char * argv[])
{
    // Get Model label and input image
    if (argc != 3)
    {
        cout << "{target} srcImg annoImg" << endl;
        return 0;
    }
    
    string srcImgFile(argv[1]);
    string annoImgFile(argv[2]);
    
    Mat srcImage = imread(srcImgFile);
    if (srcImage.empty())
    {
        cout << "Failed read source image: " << srcImgFile << endl;
        return 0;
    }
    
    if (srcImage.channels() != 1)
    {
        cvtColor(srcImage, srcImage, COLOR_RGB2GRAY);
    }
    
    int srcW = srcImage.cols;
    int srcH = srcImage.rows;
 
    int W = 5;
    float sigma = 0.1;
    Mat xxGauKernel(2 * W + 1, 2 * W + 1, CV_32FC1, Scalar::all(0));
    Mat xyGauKernel(2 * W + 1, 2 * W + 1, CV_32FC1, Scalar::all(0));
    Mat yyGauKernel(2 * W + 1, 2 * W + 1, CV_32FC1, Scalar::all(0));
 
    //构建高斯二阶偏导数模板
    for (int i = -W; i <= W;i++)
    {
        for (int j = -W; j <= W; j++)
        {
            xxGauKernel.at<float>(i + W, j + W) =
                (1 - (i*i) / (sigma*sigma))*exp(-1 * (i*i + j*j) / (2 * sigma*sigma))*(-1 / (2 * PI*pow(sigma, 4)));
            yyGauKernel.at<float>(i + W, j + W) =
                (1 - (j*j) / (sigma*sigma))*exp(-1 * (i*i + j*j) / (2 * sigma*sigma))*(-1 / (2 * PI*pow(sigma, 4)));
            xyGauKernel.at<float>(i + W, j + W) =
                ((i*j))*exp(-1 * (i*i + j*j) / (2 * sigma*sigma))*(1 / (2 * PI*pow(sigma, 6)));
        }
    }
 
    for (int i = 0; i < (2 * W + 1); i++)
    {
        for (int j = 0; j < (2 * W + 1); j++)
        {
            cout << xxGauKernel.at<float>(i, j) << "  ";
        }
        cout << endl;
    }
 
    Mat xxDerivative(srcH, srcW, CV_32FC1, Scalar::all(0));
    Mat yyDerivative(srcH, srcW, CV_32FC1, Scalar::all(0));
    Mat xyDerivative(srcH, srcW, CV_32FC1, Scalar::all(0));
    
    //图像与高斯二阶偏导数模板进行卷积
    filter2D(srcImage, xxDerivative, xxDerivative.depth(), xxGauKernel);
    filter2D(srcImage, yyDerivative, yyDerivative.depth(), yyGauKernel);
    filter2D(srcImage, xyDerivative, xyDerivative.depth(), xyGauKernel);
 
    Mat outImage(srcH, srcW, CV_32FC1, Scalar::all(0));

    Mat eValue;
    Mat eVector;
    for (int h = 0; h < srcH; h++)
    {
        for (int w = 0; w < srcW; w++)
        {
            float fxx = xxDerivative.at<float>(h, w);
            float fyy = yyDerivative.at<float>(h, w);
            float fxy = xyDerivative.at<float>(h, w);
 
            float myArray[2][2] = {{ fxx, fxy }, { fxy, fyy }};          //构建矩阵，求取特征值
 
            Mat Array(2, 2, CV_32FC1, myArray);
            eigen(Array, eValue, eVector);                               //矩阵是降序排列的
            float a1 = eValue.at<float>(0, 0);
            float a2 = eValue.at<float>(1, 0);
 
            if ((a1>0) && (ABS(a1)>(1 + ABS(a2))))             //根据特征向量判断线性结构
            {
                cout << "Found!" << endl;
                outImage.at<float>(h, w) =  pow((ABS(a1) - ABS(a2)), 4);
                //outImage.at<uchar>(h, w) = pow((ABS(a1) / ABS(a2))*(ABS(a1) - ABS(a2)), 1.5);
            }
        }
    }
 
    float maxValue = *max_element(outImage.begin<float>(), outImage.end<float>());
    float minValue = *min_element(outImage.begin<float>(), outImage.end<float>());
    cout << "maxValue: " << maxValue << endl;
    cout << "minValue: " << minValue << endl;

    //----------做一个闭操作
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 2));
    morphologyEx(outImage, outImage, MORPH_CLOSE, element);
    
    imwrite(annoImgFile, outImage);

    return 0;
}

/*
int main(int argc, const char * argv[])
{
    // Get Model label and input image
    if (argc != 3)
    {
        cout << "{target} srcImg annoImg" << endl;
        return 0;
    }
    
    double b[5][5] = {
        { 1.96 , -6.49, -0.47, -7.20, -0.65},
        { -6.49,  3.80, -6.39,  1.50, -6.34},
        { -0.47, -6.39,  4.17, -1.51,  2.67},
        { -7.20,  1.50, -1.51,  5.70,  1.80},
        { -0.65, -6.34,  2.67,  1.80, -7.10}
        };

    cv::Mat E, V;
    cv::Mat M(5,5,CV_64FC1,b);
    cv::eigen(M, E, V);
    // E: eigenvalues, 1X5, row vector

    // eigenvalues sorted in descend order
    for(int i=0; i < 5; i++)
        std::cout << E.at<double>(0,i) << " \t";
    
    cout << endl;
    return 0;
}
*/
