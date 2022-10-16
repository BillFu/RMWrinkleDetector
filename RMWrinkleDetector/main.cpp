#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "frangi_rm.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // Get Model label and input image
    if (argc != 3)
    {
        cout << "{target} srcImg annoImg" << endl;
        return 0;
    }
    
    string srcImgFile(argv[1]);
    string annoImgFile(argv[2]);
    
	//读取图片，进行处理
	Mat input_img = imread(srcImgFile, 0);
    if (input_img.empty())
    {
        cout << "Failed read source image: " << srcImgFile << endl;
        return 0;
    }
    
	Mat input_img_fl;
    
	//转换为单通道，浮点运算
	input_img.convertTo(input_img_fl, CV_32FC1);
	
    //进行处理
	Mat vesselness, scale, angles;
    
    //使用默认参数设定Frangi
    Frangi2d_Opts opts;
    Frangi2d_CreateOpts(&opts);
    
	DoFrangi2d(input_img_fl, vesselness, scale, angles, opts);
	
    //显示结果
	vesselness.convertTo(vesselness, CV_8UC1, 255);
	scale.convertTo(scale, CV_8UC1, 255);
	angles.convertTo(angles, CV_8UC1, 255);
    
	imwrite(annoImgFile, vesselness);
}
