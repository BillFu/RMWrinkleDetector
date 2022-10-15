#include <opencv2/opencv.hpp>


//options for the filter
typedef struct{
	//vessel scales
	int sigma_start;
	int sigma_end;
	int sigma_step;
	
	//BetaOne: suppression of blob-like structures. 
	//BetaTwo: background suppression. (See Frangi1998...)
	float BetaOne;
	float BetaTwo;

	bool BlackWhite; //enhance black structures if true, otherwise enhance white structures
} frangi2d_opts_t;

#define DEFAULT_SIGMA_START     3
#define DEFAULT_SIGMA_END       7
#define DEFAULT_SIGMA_STEP      1
#define DEFAULT_BETA_ONE        1.6
#define DEFAULT_BETA_TWO        0.08
#define DEFAULT_BLACKWHITE      true


/////////////////
//Frangi filter//
/////////////////

// main function
void frangi2d(const cv::Mat &src, cv::Mat &J, cv::Mat &scale, cv::Mat &directions, frangi2d_opts_t opts);

////////////////////
//Helper functions//
////////////////////

// calculate the Hessian matrix with parameter sigma on src, save to Dxx, Dxy, and Dyy
void frangi2d_hessian(const cv::Mat &src, cv::Mat &Dxx, cv::Mat &Dxy, cv::Mat &Dyy, float sigma);

// create opts: sigma_start = 3, sigma_end = 7, sigma_step = 1, Beta1 = 1.6, Beta2 = 0.08
void frangi2d_createopts(frangi2d_opts_t *opts);

// calculate the eigenvalues from Dxx, Dxy, Dyy, save result as lambda1, lambda2, Ix, Iy.
void frangi2_eig2image(const cv::Mat &Dxx, const cv::Mat &Dxy, const cv::Mat &Dyy,
                       cv::Mat &lambda1, cv::Mat &lambda2, cv::Mat &Ix, cv::Mat &Iy);
