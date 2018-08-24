#include <vector>
#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

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
  int norm;

  if (args.type==ORB_TYPE)
    {
cv::Ptr< cv::cuda::ORB >  orb = cv::cuda::ORB::create(args.orb_nfeatures, args.orb_scaleFactor, args.orb_nlevels, args.orb_edgeThreshold, args.orb_firstLevel, args.orb_WTA_K, 0, args.orb_patchSize, args.orb_fastThreshold, args.orb_blurForDescriptor);

      orb->detectAndCompute(frame0, cv::noArray(), keypoints_0, descriptors_0_GPU);
      orb->detectAndCompute(frame1, cv::noArray(), keypoints_1, descriptors_1_GPU);
      norm = cv::NORM_HAMMING;
    }
  else if (args.type==SURF_TYPE)
    {
      cv::cuda::GpuMat keypoints_0_GPU, keypoints_1_GPU;
      int pad_cols, pad_rows;
      /*This thing requires things to be a multiple of 64 for some reason*/
      pad_cols = (64 - frame0.cols % 64) % 64;
      pad_rows = (64 - frame0.rows % 64) % 64;
      cv::cuda::GpuMat frame0_surf, frame1_surf;
      cv::cuda::copyMakeBorder(frame0,frame0_surf,0,pad_rows,0,pad_cols,cv::BORDER_REFLECT_101);
      cv::cuda::copyMakeBorder(frame1,frame1_surf,0,pad_rows,0,pad_cols,cv::BORDER_REFLECT_101,0);
      cv::cuda::SURF_CUDA surf;
      surf.hessianThreshold = args.surf_hessianThreshold;
      surf.nOctaves = args.surf_nOctaves;
      surf.nOctaveLayers = args.surf_nOctaveLayers;
      surf.extended = args.surf_extended;
      surf.upright = args.surf_upright;
      surf.keypointsRatio = args.surf_keypointsRatio;
      surf(frame0_surf, cv::cuda::GpuMat(), keypoints_0_GPU, descriptors_0_GPU);
      surf(frame1_surf, cv::cuda::GpuMat(), keypoints_1_GPU, descriptors_1_GPU);
      norm = surf.defaultNorm();

      surf.downloadKeypoints(keypoints_0_GPU, keypoints_0);
      surf.downloadKeypoints(keypoints_1_GPU, keypoints_1);
      
      //   //Use k nearest neighbour matching, use two for ratio test
      // std::vector< std::vector< cv::DMatch > > matches;
      
      // cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(norm);
      // matcher -> knnMatch(descriptors_0_GPU, descriptors_1_GPU, matches,2);
      
      // std::vector< cv::DMatch > good;
      
      

      // for(int i=0; i < std::min(descriptors_1_GPU.rows - 1, (int) matches.size()); i++)
      //   {
      //     if (((int) matches[i].size()<=2 && (int) matches[i].size() > 0) && (matches[i][0].distance < args.ratio * matches[i][1].distance))
      // 	{
      // 	  good.push_back(matches[i][0]);
      // 	}
      //   }
      // cv::Mat immatches;
      // cv::drawMatches(cv::Mat(frame0), keypoints_0, cv::Mat(frame1), keypoints_1, good, immatches);
      // cv::imwrite("matches.png", immatches);
    }


  //Use k nearest neighbour matching, use two for ratio test
  std::vector< std::vector< cv::DMatch > > matches;

  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(norm);
  matcher -> knnMatch(descriptors_0_GPU, descriptors_1_GPU, matches,2);

  std::vector< cv::DMatch > good;

  

  for(int i=0; i < std::min(descriptors_1_GPU.rows - 1, (int) matches.size()); i++)
    {
      if (((int) matches[i].size()<=2 && (int) matches[i].size() > 0) && (matches[i][0].distance < args.ratio * matches[i][1].distance))
  	{
  	  good.push_back(matches[i][0]);
  	}
    }
  std::sort(good.begin(), good.end()); //Sorting means RANSAC starts with better initial approximation vs a random one.
  //Do based on how good the best one is.

  // std::vector<cv::DMatch> matches;
  // cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(norm);
  // matcher -> match(descriptors_0_GPU, descriptors_1_GPU, matches, cv::cuda::GpuMat());

  // float min_match=100, max_match=0;
  // float match_d;
  // for(int i=0; i < matches.size(); i++)
  //   {
  //     match_d = matches[i].distance;
  //     if (match_d < min_match) min_match=match_d;
  //     if (match_d > max_match) max_match=match_d;
  //   }
  // std::vector< cv::DMatch > good;

  // for(int i=0; i < std::min(descriptors_1_GPU.rows - 1, (int) matches.size()); i++)
  //   {
  //     if(matches[i].distance < min_match * args.ratio)
  // 	{
  // 	  good.push_back(matches[i]);
  // 	}
  //   }
  
  
  std::cout << "Number of features: " << good.size() << "\n";
  std::vector< cv::Point2f > points_0, points_1;
  for (int i=0; i < good.size(); i++)
    {
      points_0.push_back( keypoints_0[ good[i].queryIdx ].pt );
      points_1.push_back( keypoints_1[ good[i].trainIdx ].pt );
    }
  
  cv::Mat homo;

  homo = cv::findHomography( points_0, points_1, args.homo , args.ransac);
  std::cout << homo.at<double>(0,0) << ' ' << homo.at<double>(0,1) << ' ' << homo.at<double>(0,2) << "\n" << homo.at<double>(1,0) << ' ' << homo.at<double>(1,1) << ' ' << homo.at<double>(1,2) << "\n" << homo.at<double>(2,0) << ' ' << homo.at<double>(2,1) << ' ' << homo.at<double>(2,2) << "\n";
  homo(cv::Range(0,2),cv::Range(0,3)).copyTo(affine);

}
  
