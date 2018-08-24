#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/calib3d/calib3d.hpp>

const int SURF_TYPE=1;
const int ORB_TYPE=0;

struct FeatureArgs
{
  int orb_nfeatures;
  float orb_scaleFactor;
  int orb_nlevels;
  int orb_edgeThreshold;
  int orb_firstLevel;
  int orb_WTA_K;
  int orb_patchSize;
  int orb_fastThreshold;
  bool orb_blurForDescriptor;
  double surf_hessianThreshold;
  int surf_nOctaves;
  int surf_nOctaveLayers;
  bool surf_extended;
  float surf_keypointsRatio;
  bool surf_upright;
  int type;
  int homo;
  double ransac;
  float ratio;

  FeatureArgs()
  {
    orb_nfeatures=5000;
    orb_scaleFactor=1.2;
    orb_nlevels=8;
    orb_edgeThreshold=31;
    orb_firstLevel=0;
    orb_WTA_K=2;
    orb_patchSize=31;
    orb_fastThreshold=20;
    orb_blurForDescriptor=false;
    surf_hessianThreshold = 400;
    surf_nOctaves =4;
    surf_nOctaveLayers = 2;
    surf_extended = false;
    surf_keypointsRatio = 0.01f;
    surf_upright = false;
    type = SURF_TYPE;
    ratio = 0.8;
    homo = cv::RANSAC; //4,8,16
    ransac = 5.;
  }
};

void find_alignment(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::Mat& affine, const FeatureArgs& args);

#endif
