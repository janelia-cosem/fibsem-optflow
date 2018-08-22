#include <vector>
#include <iostream>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

#include "features.h"


void find_alignment(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::Mat& affine, const FeatureArgs& args)
{
  std::vector< cv::KeyPoint > keypoints_0, keypoints_1;
  cv::cuda::GpuMat descriptors_0_GPU, descriptors_1_GPU;
  if (args.type==ORB_TYPE)
    {
      cv::Ptr< cv::cuda::ORB >  orb = cv::cuda::ORB::create(args.orb_nfeatures, args.orb_scaleFactor, args.orb_nlevels, args.orb_edgeThreshold, args.orb_firstLevel, args.orb_WTA_K, 0, args.orb_patchSize, args.orb_fastThreshold, args.orb_blurForDescriptor);

      orb->detectAndCompute(frame0, cv::noArray(), keypoints_0, descriptors_0_GPU);
      orb->detectAndCompute(frame1, cv::noArray(), keypoints_1, descriptors_1_GPU);
    }
  else if (args.type==SURF_TYPE)
    {
      cv::cuda::SURF_CUDA surf;
      surf(frame0, cv::cuda::GpuMat(), keypoints_0, descriptors_0_GPU);
      surf(frame1, cv::cuda::GpuMat(), keypoints_1, descriptors_1_GPU);
    }


  //Use k nearest neighbour matching, use two for ratio test
  std::vector< std::vector< cv::DMatch > > matches;
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  matcher -> knnMatch(descriptors_0_GPU, descriptors_1_GPU, matches,2);

  std::vector< cv::DMatch > good;

  

  for(int i=0; i < std::min(descriptors_1_GPU.rows - 1, (int) matches.size()); i++)
    {
      if (((int) matches[i].size()<=2 && (int) matches[i].size() > 0) && (matches[i][0].distance < args.ratio * matches[i][1].distance))
	{
	  good.push_back(matches[i][0]);
	}
    }

  std::vector< cv::Point2f > points_0, points_1;
  for (int i=0; i < good.size(); i++)
    {
      points_0.push_back( keypoints_0[ good[i].queryIdx ].pt );
      points_1.push_back( keypoints_1[ good[i].trainIdx ].pt );
    }
  
  cv::Mat homo;
  std::cout << cv::RHO << " " << cv::RANSAC << "\n";
  homo = cv::findHomography( points_0, points_1, args.homo );
  std::cout << homo.at<double>(0,0) << ' ' << homo.at<double>(0,1) << ' ' << homo.at<double>(0,2) << "\n" << homo.at<double>(1,0) << ' ' << homo.at<double>(1,1) << ' ' << homo.at<double>(1,2) << "\n" << homo.at<double>(2,0) << ' ' << homo.at<double>(2,1) << ' ' << homo.at<double>(2,2) << "\n";
  homo(cv::Range(0,2),cv::Range(0,3)).copyTo(affine);
  // cv::cuda::GpuMat flow_x, flow_y;
  // cv::cuda::buildWarpAffineMaps(affine, 0, frame0.size(), flow_x, flow_y);
  // std::vector< cv::cuda::GpuMat > flow_vec;
  // flow_vec.push_back(flow_x);
  // flow_vec.push_back(flow_y);
  // cv::cuda::merge(flow_vec, flow);

}
  
