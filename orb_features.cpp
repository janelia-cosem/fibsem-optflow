#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "orb_features.h"


void find_alignment(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::Mat& affine, const OrbArgs& args)
{
  Ptr< cv::cuda::ORB >  orb = cuda::ORB::create(args.nfeatures, args.scaleFactor, args.nlevels, args.edgeThreshold, args.firstLevel, args.WTA_K, 0, args.patchSize, args.fastThreshold, args.blurForDescriptor);
  std::vector< cv::KeyPoint > keypoints_0, keypoints_1;
  cv::cuda::GpuMat descriptors_0_GPU, descriptors_1_GPU;
  orb->detectAndCompute(frame0, cv::noArray(), keypoints_0, descriptors_0_GPU);
  orb->detectAndCompute(frame1, cv::noArray(), keypoints_1, descriptors_1_GPU);

  cv::gpu::BruteForceMatcher_GPU< cv::Hamming > matcher;
  std::vector< std::vector< cv::DMatch > > matches;

  //Use k nearest neighbour matching, use two for ratio test
  matcher.knnMatch(descriptors_0_GPU, descriptors_1_GPU, matches, 2);
  std::vector< cv::DMatch > good;

  for(int i=0; i < min(descriptors_1_GPU.rows - 1, (int) matches.size()); i++)
    {
      if (((int) matches[i].size()<=2 && (int) matches[i].size() > 0) && (matches[i][0].distance < args.ratio * matches[i][1].distance))
	{
	  good.push_back(matches[i][0]);
	}
    }
  
}
  
