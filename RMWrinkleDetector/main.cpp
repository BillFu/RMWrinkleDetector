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
    
    int width = srcImage.cols;
    int height = srcImage.rows;
 
    Mat outImage(height, width, CV_8UC1,Scalar::all(0));
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
            xxGauKernel.at<float>(i + W, j + W) = (1 - (i*i) / (sigma*sigma))*exp(-1 * (i*i + j*j) / (2 * sigma*sigma))*(-1 / (2 * PI*pow(sigma, 4)));
            yyGauKernel.at<float>(i + W, j + W) = (1 - (j*j) / (sigma*sigma))*exp(-1 * (i*i + j*j) / (2 * sigma*sigma))*(-1 / (2 * PI*pow(sigma, 4)));
            xyGauKernel.at<float>(i + W, j + W) = ((i*j))*exp(-1 * (i*i + j*j) / (2 * sigma*sigma))*(1 / (2 * PI*pow(sigma, 6)));
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
 
    Mat xxDerivae(height, width, CV_32FC1, Scalar::all(0));
    Mat yyDerivae(height, width, CV_32FC1, Scalar::all(0));
    Mat xyDerivae(height, width, CV_32FC1, Scalar::all(0));
        //图像与高斯二阶偏导数模板进行卷积
    filter2D(srcImage, xxDerivae, xxDerivae.depth(), xxGauKernel);
    filter2D(srcImage, yyDerivae, yyDerivae.depth(), yyGauKernel);
    filter2D(srcImage, xyDerivae, xyDerivae.depth(), xyGauKernel);
 
 
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            
            
                //map<int, float> best_step;
                
            /*    int HLx = h - STEP; if (HLx < 0){ HLx = 0; }
                int HUx = h + STEP; if (HUx >= height){ HUx = height - 1; }
                int WLy = w - STEP; if (WLy < 0){ WLy = 0; }
                int WUy = w + STEP; if (WUy >= width){ WUy = width - 1; }
                float fxx = srcImage.at<uchar>(h, WUy) + srcImage.at<uchar>(h, WLy) - 2 * srcImage.at<uchar>(h, w);
                float fyy = srcImage.at<uchar>(HLx, w) + srcImage.at<uchar>(HUx, w) - 2 * srcImage.at<uchar>(h, w);
                float fxy = 0.25*(srcImage.at<uchar>(HUx, WUy) + srcImage.at<uchar>(HLx, WLy) - srcImage.at<uchar>(HUx, WLy) - srcImage.at<uchar>(HLx, WUy));*/
 
 
            float fxx = xxDerivae.at<float>(h, w);
            float fyy = yyDerivae.at<float>(h, w);
            float fxy = xyDerivae.at<float>(h, w);
 
 
            float myArray[2][2] = { { fxx, fxy }, { fxy, fyy } };          //构建矩阵，求取特征值
 
            Mat Array(2, 2, CV_32FC1, myArray);
            Mat eValue;
            Mat eVector;
 
            eigen(Array, eValue, eVector);                               //矩阵是降序排列的
            float a1 = eValue.at<float>(0, 0);
            float a2 = eValue.at<float>(1, 0);
 
            if ((a1>0) && (ABS(a1)>(1+ ABS(a2))))             //根据特征向量判断线性结构
            {
 
 
                outImage.at<uchar>(h, w) =  pow((ABS(a1) - ABS(a2)), 4);
                //outImage.at<uchar>(h, w) = pow((ABS(a1) / ABS(a2))*(ABS(a1) - ABS(a2)), 1.5);
                
                
            }
 
 
                
        }
 
    }
 
    
//----------做一个闭操作
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 2));
    morphologyEx(outImage, outImage, MORPH_CLOSE, element);
    
    imwrite("temp.bmp", outImage);
 
    imshow("[原始图]", outImage);
    waitKey(0);
 
    system("pause");
    return 0;
}
