#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <vector>

#include <opencv2/core/utility.hpp>
#include "opencv2/video.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>

#include "orb_features.h"
#include "optflow.h"


const std::string keys =
  "{ @output | | output flow}"
  "{ @frame0 | | frame 0}"
  "{ frame1 | | frame}"
  "{ crop | 0 | crop size}"
  "{ style | 0 | style}"
  "{ scale | 1 | scale}"
  "{ tau | | tau}"
  "{ lambda | | lambda}"
  "{ theta | | theta}"
  "{ nscales | | nscales}"
  "{ warps | | warps}"
  "{ epsilon | | epsilon}"
  "{ iterations | | iterations}"
  "{ scaleStep | | scaleStep}"
  "{ gamma | | gamma}"
  "{ top | 0 | Size of top resin}"
  "{ bottom | 0 | Size of bottom resin}"
  "{ border | 0 | border}"
  "{ orb | | Use orb features for initial flow }"
  "{ orbn | | orb nfeatures }"
  "{ orbscale | | orb scaleFactor }"
  "{ orbnlevels | | orb nlevels }"
  "{ orbedge | | orb edgeThreshold }"
  "{ orbfirst | | orb firstLevel }"
  "{ orbWTA | | orb WTA_K factor }"
  "{ orbpatch | | orb patchSize }"
  "{ orbfast | | orb fast threshold}"
  "{ orbblur | | orb blur}"
  "{ orbratio | | orb ratio}"
  "{help h || show help message }"
  ;


int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
      {
        parser.printMessage();
        return 0;
      }
    
    std::string file = parser.get<std::string>( "@output" );
    std::string frame0_name = parser.get<std::string>( "@frame0" );
    std::string frame1_name = parser.get<std::string>( "frame1" );
    int crop_width = parser.get<int>( "crop" );
    int style = parser.get<int>( "style" );
    float scale = parser.get<float>( "scale" );
    int top = parser.get<int>( "top" );
    int bottom = parser.get<int>( "bottom" );
    int border = parser.get<int>("border");
    bool orb;
    OptflowArgs args;
    OrbArgs orbargs;
    
    if (parser.has("tau")) args.tau = parser.get<double>( "tau" );
    if (parser.has("lambda"))
      {
	args.lambda = parser.get<double>( "lambda" );
      }
    else
      {
	args.lambda = args.lambda/scale;
      }
    if (parser.has("theta")) args.theta = parser.get<double>( "theta" );
    if (parser.has("nscales")) args.nscales = parser.get<int>( "nscales" );
    if (parser.has("warps")) args.warps = parser.get<int>( "warps" );
    if (parser.has("epsilon")) args.epsilon = parser.get<double>( "epsilon" );
    if (parser.has("iterations")) args.iterations = parser.get<int>( "iterations" );
    if (parser.has("scaleStep")) args.scaleStep = parser.get<double>( "scaleStep" );
    if (parser.has("gamma")) args.gamma = parser.get<double>( "gamma" );
    if (parser.has("orb"))
      {
	orb = true;
	args.useInitialFlow = true;
      }
    if (parser.has("orbn")) orbargs.nfeatures = parser.get<int>( "orbn" );
    if (parser.has("orbscale")) orbargs.scaleFactor = parser.get<float>( "orbscale" );
    if (parser.has("orbnlevels")) orbargs.nlevels = parser.get<int>( "orbnlevels" );
    if (parser.has("orbedge")) orbargs.edgeThreshold = parser.get<int>( "orbedge" );
    if (parser.has("orbfirst")) orbargs.firstLevel = parser.get<int>( "orbfirst" );
    if (parser.has("orbWTA")) orbargs.WTA_K = parser.get<int>( "orbWTA" );
    if (parser.has("orbpatch")) orbargs.patchSize = parser.get<int>( "orbpatch" );
    if (parser.has("orbfast")) orbargs.fastThreshold = parser.get<int>( "orbfast" );
    if (parser.has("orbblur")) orbargs.blurForDescriptor = true;
    if (parser.has("orbratio")) orbargs.ratio = parser.get<float>( "orbratio" );
    
    int pass_fail;

    if ( style == 0 && (frame0_name.empty() || frame1_name.empty() || file.empty()))
      {
        std::cerr << "Usage : " << argv[0] << " [<output_flow>] [<frame0>] [<frame1>]" << std::endl;
        return -1;
      }
    else if (style == 1 && (frame0_name.empty() || file.empty()))
      {
	std::cerr << "Usage : " << argv[0] << " [<output_flow>] [<frame0>] " << std::endl;
        return -1;
      }

    if ( style == 0)
      {
	pass_fail = two_file(frame0_name, frame1_name, file, crop_width, scale, top, bottom, orb, args, orbargs);
	  }
    else if (style == 1)
      {
	pass_fail = from_file(frame0_name, file, scale, top, bottom, orb, args, orbargs);
      }
    else if (style == 2)
      {
	pass_fail = average_flow(frame0_name, file, scale, border, args);
      }
    return pass_fail;
}

int two_file(std::string frame0_name, std::string frame1_name, std::string file, int crop_width, float scale, int top, int bottom, bool orb, const OptflowArgs& args, const OrbArgs& orbargs)
{


    cv::Mat frame0 = imread(frame0_name, cv::IMREAD_GRAYSCALE);
    cv::Mat frame1 = imread(frame1_name, cv::IMREAD_GRAYSCALE);
    if (frame1.size() != frame0.size())
      {
        std::cerr << "Images should be of equal sizes" << std::endl;
        return -1;
      }

    if (crop_width)
      {
	cv::Rect roi_0;
	cv::Rect roi_1;
	roi_0.x = frame0.cols - crop_width;
	roi_1.x = 0;
	roi_0.y = 0;
	roi_1.y = 0;
	roi_0.width = crop_width;
	roi_1.width = crop_width;
	roi_0.height = frame0.rows;
	roi_1.height = frame1.rows;
	frame0 = frame0(roi_0);
	frame1 = frame1(roi_1);
      }
    if (scale != 1)
      {
	resize(frame0, frame0, cv::Size(), scale, scale);
	resize(frame1, frame1, cv::Size(), scale, scale);
      }
    cv::Rect roi_top, roi_bottom;
    std::vector< cv::Rect > rois;

    roi_top.x = 0;
    roi_bottom.x = 0;
    roi_top.width = frame0.cols;
    roi_bottom.width = frame0.cols;
    roi_top.height = 0; //bottom isn't used if not flagged so don't need to initialise
    top = top*scale;
    bottom = bottom*scale;
    std::string output_dir = "";
    std::string out_name = file;
    if (!top)
      {
	rois.push_back(roi_top);
      }
    else
      {
	roi_top.y = 0;
	roi_top.height = top;
	rois.push_back(roi_top);
      }
    if (bottom)
      {
	roi_bottom.y = frame0.rows-bottom;
	roi_bottom.height= bottom;
	rois.push_back(roi_bottom);
      }
    solve_rois(frame0, frame1, output_dir, out_name, rois, orb, args, orbargs);
    
    return 0;
}

int from_file(std::string file_name, std::string output_dir, float scale, int top, int bottom, bool orb, const OptflowArgs& args, const OrbArgs& orbargs)
{
  std::ifstream infile(file_name.c_str());
  std::string frame0_name, frame1_name, out_name, old_frame0="", old_frame1="";
  cv::Mat frame0, frame1;
  top = top*scale;
  bottom = bottom*scale;
      
  while (infile >> frame0_name >> frame1_name >> out_name)
    {
      printf("%s %s\n", frame0_name.c_str(), frame1_name.c_str());
      if (frame0_name == old_frame1)
	{
	  frame0 = frame1;
	}
      else if (frame0_name != old_frame0)
	{
	  frame0 = imread(frame0_name, cv::IMREAD_GRAYSCALE);
	  if (scale != 1) resize(frame0, frame0, cv::Size(), scale, scale);
	}
      if (frame1_name == old_frame0) //These two if/elif can't both be true
	{
	  frame1 = frame0; 
	}
      else if (frame1_name != old_frame1)
	{
	  frame1 = imread(frame1_name, cv::IMREAD_GRAYSCALE);
	  if (scale != 1) resize(frame1, frame1, cv::Size(), scale, scale);
	}
      old_frame0 = frame0_name;
      old_frame1 = frame1_name;
      cv::Rect roi_top, roi_bottom;
      std::vector< cv::Rect > rois;
      
      roi_top.x = 0;
      roi_bottom.x = 0;
      roi_top.width = frame0.cols;
      roi_bottom.width = frame0.cols;
      roi_top.height = 0; //bottom isn't used if not flagged so don't need to initialise
      top = top*scale;
      bottom = bottom*scale;
      std::string output_dir = "";
      std::string out_name = file_name;
      if (!top)
	{
	  rois.push_back(roi_top);
	}
      else
	{
	  roi_top.y = 0;
	  roi_top.height = top;
	  rois.push_back(roi_top);
	}
      if (bottom)
	{
	  roi_bottom.y = frame0.rows-bottom;
	  roi_bottom.height= bottom;
	  rois.push_back(roi_bottom);
	}
      solve_rois(frame0, frame1, output_dir, out_name+"_"+std::to_string(scale), rois, orb, args, orbargs);
    }
  
  return 0;
}

int average_flow(std::string file_name, std::string output_dir, float scale, int border, const OptflowArgs& args)
{
  std::string im_name;
  std::ifstream infile(file_name.c_str());
  std::deque<cv::Mat> frames;
  cv::Mat curr_frame;
  std::vector<std::string> fnames;
  //Magic numbers (e**-x^2/4 resummed to one)
  cv::Vec6f weights(exp(-9./4.), exp(-1.), exp(-1./4.), exp(-1./4.), exp(-1.), exp(-9./4.));

  weights = weights * 0.5/(exp(-9./4.)+exp(-1.)+exp(-1./4.));

  //Read file in
  while (getline(infile,im_name))
    {
      fnames.push_back(im_name);
    }
  
  //Won't be too large
  int n_files = static_cast<int>(fnames.size());
  curr_frame = imread(fnames.at(0), cv::IMREAD_GRAYSCALE);
  
  cv::Mat blur_image(curr_frame.size(), CV_64F, cv::Scalar(0));
  cv::Mat blur_8;
  //Initialise matricies;
  frames.push_back(blur_image); //Will be popped
  frames.push_back(curr_frame);
  curr_frame.release(); //It's done its job
  frames.push_back(imread(fnames.at(1), cv::IMREAD_GRAYSCALE));
  frames.push_back(imread(fnames.at(2), cv::IMREAD_GRAYSCALE));
  frames.push_back(imread(fnames.at(3), cv::IMREAD_GRAYSCALE));
  frames.push_back(imread(fnames.at(4), cv::IMREAD_GRAYSCALE));
  frames.push_back(imread(fnames.at(5), cv::IMREAD_GRAYSCALE));
  
  for (int i=3; i<n_files-3; i++)
    {
      std::cout << "N: " << std::to_string(i) << " " << fnames.at(i) << "\n";
      frames.pop_front();
      frames.push_back(imread(fnames.at(i+3), cv::IMREAD_GRAYSCALE));
      blur_image = weights(0)*frames.at(0) + weights(1)*frames.at(1) + weights(2)*frames.at(2) + weights(3)*frames.at(4) + weights(4)*frames.at(5) + weights(5)*frames.at(6);
      blur_image.convertTo(blur_8, CV_8U, 1.);
      remap_and_save(output_dir, i, frames.at(3), blur_8, scale, border, args);
    }
      
  return 0;
}


void remap_and_save(std::string output_dir, int i, cv::Mat frame, cv::Mat blur, float scale, int border, const OptflowArgs& args)
{
  cv::cuda::GpuMat frame_GPU, blur_GPU, flow_GPU;
  cv::Mat scale_flow, flow;
  cv::Mat scale_frame;
  resize(frame, scale_frame, cv::Size(), scale, scale);
  resize(blur, blur, cv::Size(), scale, scale);
  frame_GPU.upload(scale_frame);
  blur_GPU.upload(blur);
  TVL1_solve(frame_GPU, blur_GPU, flow_GPU, args);
  flow_GPU.download(scale_flow);
  scale_flow = 1/scale * scale_flow;
  resize(scale_flow, flow, cv::Size(), 1/scale, 1/scale);

  cv::Mat new_frame(frame.rows+2*border, frame.cols+2*border, CV_8U);
  cv::Mat framed_frame;
  cv::Mat framed_flow;
  cv::copyMakeBorder(frame, framed_frame, border, border, border, border, 0);
  
  cv::copyMakeBorder(flow, framed_flow, border, border, border, border, 0);

  cv::Mat map(flow.size(), CV_32FC2);
  for (int y = 0; y < map.rows; y++)
    {
      for (int x = 0; x < map.cols; x++)
	{
	  cv::Point2f df = flow.at<cv::Point2f>(y,x);
	  map.at<cv::Point2f>(y,x) = cv::Point2f(x-df.x, y-df.y);
	}
    }
  cv::Mat framed_map;
  cv::copyMakeBorder(map, framed_map,  border, border, border, border, 0);
  std::vector<cv::Mat> map_xy;
  cv::split(framed_map, map_xy);
  cv::remap(framed_frame, new_frame, map_xy[0], map_xy[1], 1);
  cv::imwrite(output_dir+"/"+std::to_string(i)+".tiff", new_frame);
}


void solve_rois(cv::Mat frame0, cv::Mat frame1, std::string output_dir, std::string out_name, std::vector<cv::Rect> rois, bool orb, const OptflowArgs& args, const OrbArgs& orbargs)
{
  cv::cuda::GpuMat frame0_GPU, frame1_GPU;
  frame0_GPU.upload(frame0);
  frame1_GPU.upload(frame1);
  cv::cuda::GpuMat flow_GPU = frame0_GPU;

  if (orb)
    {
      find_alignment(frame0_GPU, frame1_GPU, flow_GPU, orbargs);
    }
  if ( rois.size() == 1 )
    {
      solve_wrapper(frame0_GPU, frame1_GPU, flow_GPU, output_dir, out_name, args);
    }
  else
    {
      if ( rois.at(0).height > 0)
	{
	  cv::cuda::GpuMat sub_flow;
	  sub_flow = flow_GPU(rois.at(0));
	  solve_wrapper(frame0_GPU(rois.at(0)), frame1_GPU(rois.at(0)), sub_flow, output_dir, out_name+"_top", args);
	}
      if ( rois.at(1).height > 0)
	{
	  cv::cuda::GpuMat sub_flow;
	  sub_flow = flow_GPU(rois.at(1));
	  solve_wrapper(frame0_GPU(rois.at(1)), frame1_GPU(rois.at(1)), sub_flow, output_dir, out_name+"_bottom", args);
	}
    }
}

void solve_wrapper(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::cuda::GpuMat flow_GPU, std::string output_dir, std::string out_name, const OptflowArgs& args)
{  
  TVL1_solve(frame0, frame1, flow_GPU, args);
  cv::Mat_<cv::Point2f> flow;

  flow_GPU.download(flow);
  std::string file_x = output_dir+"/"+out_name+"_x.tiff";
  std::string file_y = output_dir+"/"+out_name+"_y.tiff";
  std::vector<cv::Mat> flow_xy;
  cv::split(flow, flow_xy);
  
  imwrite(file_x, flow_xy[0]);
  imwrite(file_y, flow_xy[1]);
}

void TVL1_solve(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::cuda::GpuMat& output, const OptflowArgs& args)
{
  cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> solver = cv::cuda::OpticalFlowDual_TVL1::create(args.tau, args.lambda, args.theta, args.nscales, args.warps, args.epsilon, args.iterations, args.scaleStep, args.gamma);
  solver -> calc(frame0, frame1, output);
}
