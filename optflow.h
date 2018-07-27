#ifndef OPTFLOW_H
#define OPTFLOW_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace std;
using namespace cv::cuda;


struct OptflowArgs
{
  double tau;
  double lambda;
  double theta;
  int nscales;
  int warps;
  double epsilon;
  int iterations;
  double scaleStep;
  double gamma;
  bool useInitialFlow;
  OptflowArgs()
  {
  tau = 0.25;
  lambda = 0.05;
  theta = 0.3;
  nscales = 5;
  warps = 5;
  epsilon = 0.01;
  iterations = 300;
  scaleStep = 0.8;
  gamma = 0.0;
  useInitialFlow = false;
  }
};
  

int two_file(string frame0_name, string frame1_name, string file, int crop_width, float scale, int top, int bottom, const OptflowArgs& args);

int from_file(string file_name, string output_dir, float scale, int top, int bottom, const OptflowArgs& args);

void solve_wrapper(Mat frame0, Mat frame1, string output_dir, string out_name, const OptflowArgs& args);

void TVL1_solve(GpuMat frame0, GpuMat frame1, GpuMat& output, const OptflowArgs& args); 

#endif
