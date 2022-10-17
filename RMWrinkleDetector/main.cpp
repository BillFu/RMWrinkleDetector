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
	//Mat srcImg = imread(srcImgFile, 0); // 0 is IMREAD_GRAYSCALE
    Mat srcImgBGR = imread(srcImgFile);
    if (srcImgBGR.empty())
    {
        cout << "Failed read source image: " << srcImgFile << endl;
        return 0;
    }
    
    Mat hsvImg;
    cvtColor(srcImgBGR, hsvImg, COLOR_BGR2HSV);
    
    vector<Mat> channels(3);
    // split img:
    split(hsvImg, channels);
    // get the channels (dont forget they follow BGR order in OpenCV)
    
    imwrite("case1_v.png", channels[2]);
    imwrite("case1_s.png", channels[1]);
    imwrite("case1_h.png", channels[0]);

    /*
	Mat input_img_fl;
	//转换为单通道，浮点运算
    srcImg.convertTo(input_img_fl, CV_32FC1);
    
    cout << "Channels in srcImg after convertT0() : " << srcImg.channels() << endl;
    
    //imwrite("case1_gray.png", input_img_fl);
    */
    /*
    Mat grayInImage;
    cvtColor(input_img, grayInImage, COLOR_BGR2GRAY);
	
    */
    /*

    //进行处理
	Mat Vmap, scale, angles;
    
    //使用默认参数设定Frangi
    Frangi2d_Opts opts;
    Frangi2d_CreateOpts(&opts);
    
	DoFrangi2d(input_img_fl, Vmap, scale, angles, opts, 0.4);
	
    //显示结果
    Vmap.convertTo(Vmap, CV_8UC1, 255);
	scale.convertTo(scale, CV_8UC1, 255);
	angles.convertTo(angles, CV_8UC1, 255);
    
	imwrite(annoImgFile, Vmap);
    */
}
