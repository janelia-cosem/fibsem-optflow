#ifndef ORB_FEATURES_H
#define ORB_FEATURES_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>

struct OrbArgs
{
  int nfeatures;
  float scaleFactor;
  int nlevels;
  int edgeThreshold;
  int firstLevel;
  int WTA_K;
  int patchSize;
  int fastThreshold;
  bool blurForDescriptor;
  float ratio;
  OrbArgs()
  {
    nfeatures=500;
    scaleFactor=1.2;
    nlevels=8;
    edgeThreshold=31;
    firstLevel=0;
    WTA_K=2;
    patchSize=31;
    fastThreshold=20;
    blurForDescriptor=false;
    ratio = 0.92;
  }
};

void find_alignment(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::Mat& affine, const OrbArgs& args)

#endif
