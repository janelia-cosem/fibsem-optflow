#ifndef OPTFLOW_H
#define OPTFLOW_H

#include <vector>
#include <string>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>

#include "orb_features.h"


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
  nscales = 10;
  warps = 5;
  epsilon = 0.01;
  iterations = 300;
  scaleStep = 0.8;
  gamma = 0.0;
  useInitialFlow = false;
  }
};
  

int two_file(std::string frame0_name, std::string frame1_name, std::string file, int crop_width, float scale, int top, int bottom, bool orb, const OptflowArgs& args, const OrbArgs& orbargs);

int from_file(std::string file_name, std::string output_dir, float scale, int top, int bottom, bool orb, const OptflowArgs& args, const OrbArgs& orbargs);

int average_flow(std::string file_name, std::string output_dir, float scale, int border, const OptflowArgs& args);

void remap_and_save(std::string output_dir, int i, cv::Mat frame, cv::Mat blur, float scale, int border, const OptflowArgs& args, const OrbArgs& orbargs);

void solve_rois(cv::Mat frame0, cv::Mat frame1, std::string output_dir, std::string out_name, std::vector<cv::Rect> rois, const OptflowArgs& args, const OrbArgs& orbargs);

void solve_wrapper(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::cuda::GpuMat flow, std::string output_dir, std::string out_name, const OptflowArgs& args);


void TVL1_solve(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::cuda::GpuMat& output, const OptflowArgs& args); 

#endif
