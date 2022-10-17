//
//  frangi_rm.h
//
//
/*

The main idea is presented in this paper: Evaluation of Automatic Facial Wrinkle Detection Algorithms
 
the reference source code from: https://www.zhihu.com/people/jsxyhelu/posts?page=1
Hessian矩阵以及在血管增强中的应用—OpenCV实现
 
Author: Fu Xiaoqiang
Date:   2022/10/15
*/

#ifndef FRANGI_RM_HPP
#define FRANGI_RM_HPP

#include <opencv2/opencv.hpp>


//options for the filter
struct Frangi2d_Opts{
	//vessel scales
	int sigma_start;
	int sigma_end;
	int sigma_step;
	
	//BetaOne: suppression of blob-like structures. 
	//BetaTwo: background suppression. (See Frangi1998...)
	float BetaOne;
	float BetaTwo;

    //enhance black structures if true, otherwise enhance white structures
    bool DarkStructBriBg;
} ;

#define DEFAULT_SIGMA_START     5
#define DEFAULT_SIGMA_END       15
#define DEFAULT_SIGMA_STEP      5

#define DEFAULT_BETA_ONE        1.6
#define DEFAULT_BETA_TWO        0.08
#define DEFAULT_DSBB            true  // DarkStructBriBg


/////////////////
//Frangi filter//
/////////////////

// main function
void DoFrangi2d(const cv::Mat &src, cv::Mat &J,
              cv::Mat &scale, cv::Mat &directions, Frangi2d_Opts opts,
                float T);

////////////////////
//Helper functions//
////////////////////

// calculate the Hessian matrix with parameter sigma on src, save to Dxx, Dxy, and Dyy
void Frangi2d_CalcHessian(const cv::Mat &src, cv::Mat &Dxx, cv::Mat &Dxy, cv::Mat &Dyy, float sigma);

// create opts: sigma_start = 3, sigma_end = 7, sigma_step = 1, Beta1 = 1.6, Beta2 = 0.08
void Frangi2d_CreateOpts(Frangi2d_Opts *opts);

// calculate the eigenvalues from Dxx, Dxy, Dyy, save result as lambda1, lambda2, Ix, Iy.
void Frangi2_Eig2Image(const cv::Mat &Dxx, const cv::Mat &Dxy, const cv::Mat &Dyy,
                       cv::Mat &lambda1, cv::Mat &lambda2, cv::Mat &Ix, cv::Mat &Iy);


#endif /* end of FRANGI_RM_HPP */
