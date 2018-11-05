#ifndef OPTFLOW_H
#define OPTFLOW_H

#include <vector>
#include <string>

#include <jsoncpp/json/json.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

 

int from_file(Json::Value& args);

/* int average_flow(std::string file_name, std::string output_dir, float scale, int border, const OptflowArgs& args); */

/* void remap_and_save(std::string output_dir, int i, cv::Mat frame, cv::Mat blur, float scale, int border, const OptflowArgs& args); */

void get_rois(Json::Value& rois, Json::Value& args, int rows, int cols);

cv::Rect roi_from_array(Json::Value& roi_array);

void solve_rois(cv::Mat& frame0, cv::Mat& frame1, Json::Value& rois, Json::Value& im_args, Json::Value& args);

void solve_wrapper(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::Mat affine, Json::Value& im_args, Json::Value& args, bool features, std::vector < cv:: Rect > roi_vec);

Json::Value generate_TV_args(const Json::Value& im_args, const Json::Value& args);

void TVL1_solve(cv::cuda::GpuMat& frame0, cv::cuda::GpuMat& frame1, cv::cuda::GpuMat& output, const Json::Value& args); 

void random_points(cv::Mat& flow_x, cv::Mat& flow_y, Json::Value& im_args, Json::Value& args, std::vector < cv::Rect > roi_vec, cv::Mat mask, bool features);

void upload_points(const Json::Value& im_args, const Json::Value& args);
#endif
