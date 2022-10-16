//
//  frangi_rm.cpp
//
//
/*********************************************************************************************
The main idea is presented in this paper:
    Evaluation of Automatic Facial Wrinkle Detection Algorithms
 
the reference source code from: https://www.zhihu.com/people/jsxyhelu/posts?page=1
Hessian矩阵以及在血管增强中的应用—OpenCV实现
 
Author: Fu Xiaoqiang
Date:   2022/10/15
 *********************************************************************************************/

#include <iostream>
#include <fstream>

#include "frangi_rm.hpp"

using namespace std;
using namespace cv;

void Frangi2d_CalcHessian(const Mat &src, Mat &Dxx, Mat &Dxy, Mat &Dyy, float sigma)
{
	// -----------------------construct Hessian kernels----------------------------
	int n_kern_x = 2*round(3*sigma) + 1;
	int n_kern_y = n_kern_x;
	float *kern_xx_f = new float[n_kern_x*n_kern_y]();
	float *kern_xy_f = new float[n_kern_x*n_kern_y]();
	float *kern_yy_f = new float[n_kern_x*n_kern_y]();
	int i=0, j=0;
    float sigmaSq = sigma*sigma;
    float sigmaPower4 = sigmaSq * sigmaSq;
    float sigmaPower6 = sigmaSq * sigmaPower4;
	for (int x = -round(3*sigma); x <= round(3*sigma); x++)
    {
		j=0;
		for (int y = -round(3*sigma); y <= round(3*sigma); y++)
        {
			kern_xx_f[i*n_kern_y + j] = 1.0f/(2.0f * M_PI * sigmaPower4) * (x*x/sigmaSq - 1) *
            exp(-(x*x + y*y)/(2.0f*sigmaSq));
			kern_xy_f[i*n_kern_y + j] = 1.0f/(2.0f*M_PI* sigmaPower6)*(x*y)*exp(-(x*x + y*y)/(2.0f*sigmaSq));
			j++;
		}
		i++;
	}
    
	for (int j=0; j < n_kern_y; j++)
    {
		for (int i=0; i < n_kern_x; i++)
        {
			kern_yy_f[j*n_kern_x + i] = kern_xx_f[i*n_kern_x + j];
		}
	}

	//flip kernels since kernels aren't symmetric
    // and opencv's filter2D operation performs a correlation, not a convolution
	Mat kern_xx;
	flip(Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xx_f), kern_xx, -1);
	
	Mat kern_xy;
	flip(Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xy_f), kern_xy, -1);

	Mat kern_yy;
	flip(Mat(n_kern_y, n_kern_x, CV_32FC1, kern_yy_f), kern_yy, -1);
    
    // -----------------------end of construct Hessian kernels----------------------------

	//specify anchor since we are to perform a convolution, not a correlation
	Point anchor(n_kern_x - n_kern_x/2 - 1, n_kern_y - n_kern_y/2 - 1);

	//run image filter
	filter2D(src, Dxx, -1, kern_xx, anchor);
	filter2D(src, Dxy, -1, kern_xy, anchor);
	filter2D(src, Dyy, -1, kern_yy, anchor);

	delete [] kern_xx_f;
	delete [] kern_xy_f;
	delete [] kern_yy_f;
}

void Frangi2d_CreateOpts(Frangi2d_Opts* opts)
{
	//these parameters depend on the scale of the vessel, depending ultimately on the image size...
	opts->sigma_start = DEFAULT_SIGMA_START;
	opts->sigma_end = DEFAULT_SIGMA_END;
	opts->sigma_step = DEFAULT_SIGMA_STEP;

	opts->BetaOne = DEFAULT_BETA_ONE; //ignore blob-like structures?
	opts->BetaTwo = DEFAULT_BETA_TWO; //appropriate background suppression for this specific image, but can change. 

	opts->BlackWhite = DEFAULT_BLACKWHITE; 
}

//estimate eigenvalues from Dxx, Dxy, Dyy. Save results to lambda1, lambda2, Ix, Iy. 
void frangi2_eig2image(const Mat &Dxx, const Mat &Dxy, const Mat &Dyy,
                       Mat &lambda1, Mat &lambda2, Mat &Ix, Mat &Iy)
{
	//calculate eigenvectors of J, v1 and v2
	Mat tmp, tmp2;
	tmp2 = Dxx - Dyy;
	sqrt(tmp2.mul(tmp2) + 4*Dxy.mul(Dxy), tmp);
	Mat v2x = 2*Dxy;
	Mat v2y = Dyy - Dxx + tmp;

	// --------------- normalize ----------------
	Mat mag;
	sqrt((v2x.mul(v2x) + v2y.mul(v2y)), mag);
	Mat v2xtmp = v2x.mul(1.0f/mag);
	v2xtmp.copyTo(v2x, mag != 0);
	Mat v2ytmp = v2y.mul(1.0f/mag);
	v2ytmp.copyTo(v2y, mag != 0);

	// ------ eigenvectors are orthogonal -------
	Mat v1x, v1y;  // ---- eigenvector
	v2y.copyTo(v1x);
	v1x = -1*v1x;
	v2x.copyTo(v1y);

	// ------------compute eigenvalues-------------
	Mat mu1 = 0.5*(Dxx + Dyy + tmp);
	Mat mu2 = 0.5*(Dxx + Dyy - tmp);

	//sort eigenvalues by absolute value abs(Lambda1) < abs(Lamda2)
	Mat check = abs(mu1) > abs(mu2);
	mu1.copyTo(lambda1);
    mu2.copyTo(lambda1, check);
	mu2.copyTo(lambda2);
    mu1.copyTo(lambda2, check);

	v1x.copyTo(Ix);
    v2x.copyTo(Ix, check);
	v1y.copyTo(Iy);
    v2y.copyTo(Iy, check);
}

//Vesselness is saved in J, scale is saved to scale, vessel angle is saved to directions.
void DoFrangi2d(const Mat &src, Mat &maxVals, Mat &whatScale,
              Mat &outAngles, Frangi2d_Opts opts)
{
	vector<Mat> AllFiltered; //All means the results in all space scales (i.e., sigma) will be collected.
	vector<Mat> AllAngles;
	float beta = 2*opts.BetaOne*opts.BetaOne;
	float c = 2*opts.BetaTwo*opts.BetaTwo;

	for (float sigma = opts.sigma_start; sigma <= opts.sigma_end; sigma += opts.sigma_step)
    {
		Mat Dxx, Dyy, Dxy;
        Frangi2d_CalcHessian(src, Dxx, Dxy, Dyy, sigma);

		Dxx = Dxx*sigma*sigma;
		Dyy = Dyy*sigma*sigma;
		Dxy = Dxy*sigma*sigma;
	
		//calculate (abs sorted) eigenvalues and vectors
		Mat lambda1, lambda2, Ix, Iy;
		frangi2_eig2image(Dxx, Dxy, Dyy, lambda1, lambda2, Ix, Iy);
		
		Mat angles;
        // phase函数计算方向场，该函数参数angleInDegrees默认为false，即弧度，当置为true时，则输出为角度。
        // 值域为0~360degrees，或者0~2*M_PI
        // 和atan2()的值域不同，-M_PI ~ M_PI
		phase(Ix, Iy, angles);
        AllAngles.push_back(angles);
		
        //Returns the next representable value after x in the direction of y.
        // nextafterf(0, 1): first representable value greater than zero: 4.940656e-324
        //Sets all or some of the array elements to the specified value.
        // the second argument is the mask
        // so, this statement intends to assign a tiny and positive value where a pixel equals zero
		lambda2.setTo(nextafterf(0, 1), lambda2 == 0);
		Mat Rb = lambda1.mul(1.0/lambda2);
		Rb = Rb.mul(Rb);
		Mat S2 = lambda1.mul(lambda1) + lambda2.mul(lambda2);

		Mat tmp1, tmp2;
		exp(-Rb/beta, tmp1);
		exp(-S2/c, tmp2);
	
        Mat I = Mat::ones(src.rows, src.cols, src.type());
		Mat Ifiltered = tmp1.mul(I - tmp2);
		if (opts.BlackWhite) // enhance black structures if true, otherwise enhance white structures
        {
			Ifiltered.setTo(0, lambda2 < 0);
		}
        else
        {
			Ifiltered.setTo(0, lambda2 > 0);
		}

		//store results
        AllFiltered.push_back(Ifiltered);
	}

	float sigma = opts.sigma_start;
    AllFiltered[0].copyTo(maxVals);
    AllFiltered[0].copyTo(whatScale);
    AllFiltered[0].copyTo(outAngles);
	whatScale.setTo(sigma);

    // 将多个尺度下的结果“压扁”在一起
	for (int i=1; i < AllFiltered.size(); i++)
    {
		maxVals = max(maxVals, AllFiltered[i]);
		whatScale.setTo(sigma, AllFiltered[i] == maxVals);
        AllAngles[i].copyTo(outAngles, AllFiltered[i] == maxVals);
		sigma += opts.sigma_step;
	}
}
