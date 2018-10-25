#include <vector>
#include <iostream>
#include <vector>

#include <jsoncpp/json/json.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

#include "features.h"

Json::Value orb_defaults(const Json::Value& im_args, const Json::Value& args)
{
  Json::Value orb_args;
  orb_args["nfeatures"] = im_args.get("nfeatures",args.get("nfeatures", 5000).asInt()).asInt();
  orb_args["scaleFactor"] = im_args.get("scaleFactor",args.get("scaleFactor", 1.2).asFloat()).asFloat();
  orb_args["nlevels"] = im_args.get("nlevels",args.get("nlevels", 8).asInt()).asInt();
  orb_args["edgeThreshold"] = im_args.get("edgeThreshold",args.get("edgeThreshold", 31).asInt()).asInt();
  orb_args["firstLevel"] = im_args.get("firstLevel",args.get("firstLevel", 0).asInt()).asInt();
  orb_args["WTA_K"] = im_args.get("WTA_K",args.get("WTA_K", 2).asInt()).asInt();
  orb_args["patchSize"] = im_args.get("patchSize",args.get("patchSize", 31).asInt()).asInt();
  orb_args["fastThreshold"] = im_args.get("fastThreshold",args.get("fastThreshold", 20).asInt()).asInt();
  orb_args["blurForDescriptor"] = im_args.get("blurForDescriptor",args.get("blurForDescriptor", false).asBool()).asBool();
  return orb_args;
}

Json::Value surf_defaults(const Json::Value& im_args, const Json::Value& args)
{
  Json::Value surf_args;
  surf_args["hessianThreshold"] = im_args.get("hessianThreshold",args.get("hessianThreshold", 400).asInt()).asInt();
  surf_args["nOctaves"] = im_args.get("nOctaves",args.get("nOctaves", 4).asInt()).asInt();
  surf_args["nOctaveLayers"] = im_args.get("nOctaveLayers",args.get("nOctaveLayers", 2).asInt()).asInt();
  surf_args["extended"] = im_args.get("extended",args.get("extended", false).asBool()).asBool();
  surf_args["keypointsRatio"] = im_args.get("keypointsRatio",args.get("keypointsRatio", 0.01).asFloat()).asFloat();
  surf_args["upright"] = im_args.get("upright",args.get("upright", false).asBool()).asBool();
  return surf_args;
}

void find_alignment(cv::cuda::GpuMat& frame0, cv::cuda::GpuMat& frame1, cv::Mat& affine, Json::Value& im_args, Json::Value& args)
{

  bool debug=args.get("debug",false).asBool();
  std::vector< cv::KeyPoint > keypoints_0, keypoints_1;
 cv::cuda::GpuMat descriptors_0_GPU, descriptors_1_GPU;
  int norm;
  int feature_type = im_args.get("features",args.get("features",SURF_TYPE).asInt()).asInt();
  if (feature_type==ORB_TYPE)
    {
      Json::Value orb_args;
      orb_args = orb_defaults(im_args, args);
      cv::Ptr< cv::cuda::ORB >  orb = cv::cuda::ORB::create(orb_args["nfeatures"].asInt(), orb_args["scaleFactor"].asDouble(), orb_args["nlevels"].asInt(), orb_args["edgeThreshold"].asInt(), orb_args["firstLevel"].asInt(), orb_args["WTA_K"].asInt(), 0, orb_args["patchSize"].asInt(), orb_args["fastThreshold"].asInt(), orb_args["blurForDescriptor"].asBool());

      orb->detectAndCompute(frame0, cv::noArray(), keypoints_0, descriptors_0_GPU);
      orb->detectAndCompute(frame1, cv::noArray(), keypoints_1, descriptors_1_GPU);
      norm = cv::NORM_HAMMING;
    }
  else if (feature_type==SURF_TYPE)
    {
      Json::Value surf_args;
      surf_args = surf_defaults(im_args, args);
      cv::cuda::GpuMat keypoints_0_GPU, keypoints_1_GPU;
      int pad_cols_0, pad_rows_0, pad_cols_1, pad_rows_1;
      /*This thing requires things to be a multiple of 64 for some reason*/
      pad_cols_0 = (64 - frame0.cols % 64) % 64;
      pad_rows_0 = (64 - frame0.rows % 64) % 64;
      pad_cols_1 = (64 - frame1.cols % 64) % 64;
      pad_rows_1 = (64 - frame1.rows % 64) % 64;

      cv::cuda::GpuMat frame0_surf, frame1_surf;
      cv::cuda::copyMakeBorder(frame0,frame0_surf,0,pad_rows_0,0,pad_cols_0,cv::BORDER_REFLECT_101);
      cv::cuda::copyMakeBorder(frame1,frame1_surf,0,pad_rows_1,0,pad_cols_1,cv::BORDER_REFLECT_101);
      cv::cuda::SURF_CUDA surf;
      surf.hessianThreshold = surf_args["hessianThreshold"].asInt();
      surf.nOctaves = surf_args["nOctaves"].asInt();
      surf.nOctaveLayers = surf_args["nOctaveLayers"].asInt();
      surf.extended = surf_args["extended"].asBool();
      surf.upright = surf_args["upright"].asBool();
      surf.keypointsRatio = surf_args["keypointsRatio"].asFloat();
      surf(frame0_surf, cv::cuda::GpuMat(), keypoints_0_GPU, descriptors_0_GPU);
      surf(frame1_surf, cv::cuda::GpuMat(), keypoints_1_GPU, descriptors_1_GPU);
      
      norm = surf.defaultNorm();

      surf.downloadKeypoints(keypoints_0_GPU, keypoints_0);
      surf.downloadKeypoints(keypoints_1_GPU, keypoints_1);
      
    }


  //Use k nearest neighbour matching, use two for ratio test
  std::vector< std::vector< cv::DMatch > > matches;

  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(norm);
  matcher -> knnMatch(descriptors_0_GPU, descriptors_1_GPU, matches,2);

  std::vector< cv::DMatch > good;

  

  for(int i=0; i < std::min(descriptors_1_GPU.rows - 1, (int) matches.size()); i++)
    {
      if (((int) matches[i].size()<=2 && (int) matches[i].size() > 0) && (matches[i][0].distance < im_args.get("ratio",args.get("ratio",0.8).asFloat()).asFloat() * matches[i][1].distance))
  	{
  	  good.push_back(matches[i][0]);
  	}
    }
  std::sort(good.begin(), good.end()); 
  if (debug)
    {
      std::cout << "Number of features: " << matches.size() << "\n";
      std::cout << "Number of good features: " << good.size() << "\n";
    }
  std::vector< cv::Point2f > points_0, points_1;
  for (int i=0; i < good.size(); i++)
    {
      points_0.push_back( keypoints_0[ good[i].queryIdx ].pt );
      points_1.push_back( keypoints_1[ good[i].trainIdx ].pt );
    }
  
  cv::Mat homo;

  homo = cv::findHomography( points_0, points_1, im_args.get("homo",args.get("homo",cv::RANSAC).asInt()).asInt() , im_args.get("ransac",args.get("ransac",5).asDouble()).asDouble());
  if (debug)
    {
      std::cout << homo.at<double>(0,0) << ' ' << homo.at<double>(0,1) << ' ' << homo.at<double>(0,2) << "\n" << homo.at<double>(1,0) << ' ' << homo.at<double>(1,1) << ' ' << homo.at<double>(1,2) << "\n" << homo.at<double>(2,0) << ' ' << homo.at<double>(2,1) << ' ' << homo.at<double>(2,2) << "\n";
    }
  homo(cv::Range(0,2),cv::Range(0,3)).copyTo(affine);

}
  
