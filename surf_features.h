#ifndef SURF_FEATURES_H
#define SURF_FEATURES_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/calib3d/calib3d.hpp>

struct SurfArgs
{
  double _hessianThreshold;
  int _nOctaves;
  int _nOctaveLayers;
  bool _extended;
  float _keypointsRatio;
  bool _upright;
  float ratio;
  int homo;
  SurfArgs()
  {
    _hessianThreshold = 400;
    _nOctaves =4;
    _nOctaveLayers = 2;
    _extended = false;
    _keypointsRatio = 0.01;
    _upright = false;
    ratio = 0.8;
    homo = cv::RANSAC; //4,8,16
  }
};

void find_alignment_surf(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::Mat& affine, const SurfArgs& args);

#endif
