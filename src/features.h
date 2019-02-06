#ifndef FEATURES_H
#define FEATURES_H

#include <jsoncpp/json/json.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/calib3d/calib3d.hpp>

const int SURF_TYPE=2;
const int ORB_TYPE=1;

Json::Value orb_defaults(const Json::Value& im_args, const Json::Value& args);

Json::Value surf_defaults(const Json::Value& im_args, const Json::Value& args);

void find_alignment(cv::cuda::GpuMat& frame0, cv::cuda::GpuMat& frame1, cv::Mat& affine, bool& affine_found, Json::Value& im_args, Json::Value& args);

#endif
