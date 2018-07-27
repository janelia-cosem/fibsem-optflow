#include <iostream>
#include <fstream>

#include <opencv2/core/utility.hpp>
#include "opencv2/video.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>

#include "optflow.h"

using namespace cv;
using namespace std;
using namespace cv::cuda;

const string keys =
  "{ @output | | output flow}"
  "{ @frame0 | | frame 0}"
  "{ frame1 | | frame}"
  "{ crop | 0 | crop size}"
  "{ style | 0 | style}"
  "{ scale | 0.5 | scale}"
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
  "{help h || show help message}"
  ;


int two_file(string frame0_name, string frame1_name, string file, int crop_width, float scale, int top, int bottom, const OptflowArgs& args)
{


    Mat frame0 = imread(frame0_name, IMREAD_GRAYSCALE);
    Mat frame1 = imread(frame1_name, IMREAD_GRAYSCALE);
    
    if (frame1.size() != frame0.size())
      {
        cerr << "Images should be of equal sizes" << endl;
        return -1;
      }

    if (crop_width)
      {
	Rect roi_0;
	Rect roi_1;
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
	resize(frame0, frame0, Size(), scale, scale);
	resize(frame1, frame1, Size(), scale, scale);
      }
    Rect roi;
    roi.x = 0;
    roi.width = frame0.cols;
    top = top*scale;
    bottom = bottom*scale;
    string output_dir = "";
    string out_name = file;
      if (!( top || bottom))
	{
	  solve_wrapper(frame0, frame1, output_dir, out_name, args);
	}
      if (top)
	{
	  roi.y = 0;
	  roi.height = top;
	  solve_wrapper(frame0(roi), frame1(roi), output_dir, out_name+"_top", args);
	}
      if (bottom)
	{
	  roi.y = frame0.rows-bottom;
	  roi.height= bottom;
	  solve_wrapper(frame0(roi), frame1(roi), output_dir, out_name+"_bottom", args);
	}

    
    return 0;
}

int from_file(string file_name, string output_dir, float scale, int top, int bottom, const OptflowArgs& args)
{
  ifstream infile(file_name.c_str());
  string frame0_name, frame1_name, out_name, old_frame0="", old_frame1="";
  Mat frame0, frame1;
  Rect roi;
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
	  frame0 = imread(frame0_name, IMREAD_GRAYSCALE);
	  if (scale != 1) resize(frame0, frame0, Size(), scale, scale);
	}
      if (frame1_name == old_frame0) //These two if/elif can't both be true
	{
	  frame1 = frame0; 
	}
      else if (frame1_name != old_frame1)
	{
	  frame1 = imread(frame1_name, IMREAD_GRAYSCALE);
	  if (scale != 1) resize(frame1, frame1, Size(), scale, scale);
	}
      old_frame0 = frame0_name;
      old_frame1 = frame1_name;

      roi.x = 0;
      roi.width = frame0.cols;
      if (!( top || bottom))
	{
	  solve_wrapper(frame0, frame1, output_dir, out_name, args);
	}
      if (top)
	{
	  roi.y = 0;
	  roi.height = top; 
	  solve_wrapper(frame0(roi), frame1(roi), output_dir, out_name+"_top", args);
	}
      if (bottom)
	{
	  roi.y = frame0.rows-bottom;
	  roi.height= bottom;
	  solve_wrapper(frame0(roi), frame1(roi), output_dir, out_name+"_bottom", args);
	}
    }

  return 0;
}

  
int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
      {
        parser.printMessage();
        return 0;
      }
    
    string file = parser.get<string>( "@output" );
    string frame0_name = parser.get<string>( "@frame0" );
    string frame1_name = parser.get<string>( "frame1" );
    int crop_width = parser.get<int>( "crop" );
    int style = parser.get<int>( "style" );
    float scale = parser.get<float>( "scale" );
    int top = parser.get<int>( "top" );
    int bottom = parser.get<int>( "bottom" );
    OptflowArgs args;
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
    
    int pass_fail;

    if ( style == 0 && (frame0_name.empty() || frame1_name.empty() || file.empty()))
      {
        cerr << "Usage : " << argv[0] << " [<output_flow>] [<frame0>] [<frame1>]" << endl;
        return -1;
      }
    else if (style == 1 && (frame0_name.empty() || file.empty()))
      {
	cerr << "Usage : " << argv[0] << " [<output_flow>] [<frame0>] " << endl;
        return -1;
      }

    if ( style == 0)
      {
	pass_fail = two_file(frame0_name, frame1_name, file, crop_width, scale, top, bottom, args);
	  }
    else if (style == 1)
      {
	pass_fail = from_file(frame0_name, file, scale, top, bottom, args);
      }
    return pass_fail;
}

void solve_wrapper(Mat frame0, Mat frame1, string output_dir, string out_name, const OptflowArgs& args)
{
  GpuMat frame0_GPU, frame1_GPU, flow_GPU;
  frame0_GPU.upload(frame0);
  frame1_GPU.upload(frame1);
  Mat_<Point2f> flow;
  TVL1_solve(frame0_GPU, frame1_GPU, flow_GPU, args);
  flow_GPU.download(flow);
  string file_x = output_dir+"/"+out_name+"_x.tiff";
  string file_y = output_dir+"/"+out_name+"_y.tiff";
  vector<Mat> flow_xy;
  split(flow, flow_xy);
  
  imwrite(file_x, flow_xy[0]);
  imwrite(file_y, flow_xy[1]);
}

void TVL1_solve(GpuMat frame0, GpuMat frame1, GpuMat& output, const OptflowArgs& args)
{
  Ptr<OpticalFlowDual_TVL1> solver = cv::cuda::OpticalFlowDual_TVL1::create(args.tau, args.lambda, args.theta, args.nscales, args.warps, args.epsilon, args.iterations, args.scaleStep, args.gamma);
  solver -> calc(frame0, frame1, output);
}
