#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "features.h"
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
  "{ feature | | type of feature }"
  "{ template | | use template matching}"
  "{ temp_meth | | template method}"
  "{ orbn | | orb nfeatures }"
  "{ orbscale | | orb scaleFactor }"
  "{ orbnlevels | | orb nlevels }"
  "{ orbedge | | orb edgeThreshold }"
  "{ orbfirst | | orb firstLevel }"
  "{ orbWTA | | orb WTA_K factor }"
  "{ orbpatch | | orb patchSize }"
  "{ orbfast | | orb fast threshold}"
  "{ orbblur | | orb blur}"
  "{ ratio | | feature ratio}"
  "{ homo | | feature homography method}"
  "{ ransac | | ransac threshold}"
  "{ surfhess | | surf hessianthreshold}"
  "{ surfoct | | surf octaves}"
  "{ surfoctL | | surf octave layers}"
  "{ surfext | | surf extended}"
  "{ surfkey | | surf keypoints ratio}"
  "{ surfup | | surf upright?}"
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
    bool use_template = false;
    bool features=false;
    OptflowArgs args;
    FeatureArgs featureargs;
    
    if (parser.has("tau")) args.tau = parser.get<double>( "tau" );
    if (parser.has("lambda")) args.lambda = parser.get<double>( "lambda" );
    if (parser.has("theta")) args.theta = parser.get<double>( "theta" );
    if (parser.has("nscales")) args.nscales = parser.get<int>( "nscales" );
    if (parser.has("warps")) args.warps = parser.get<int>( "warps" );
    if (parser.has("epsilon")) args.epsilon = parser.get<double>( "epsilon" );
    if (parser.has("iterations")) args.iterations = parser.get<int>( "iterations" );
    if (parser.has("scaleStep")) args.scaleStep = parser.get<double>( "scaleStep" );
    if (parser.has("gamma")) args.gamma = parser.get<double>( "gamma" );
    if (parser.has("template"))	use_template=true;
    if (parser.has("temp_meth")) args.temp_method = parser.get<int>( "temp_method" );
    if (parser.has("feature"))
      {
	featureargs.type = parser.get<int>( "feature" );
	features = true;
      }
    if (parser.has("orbn")) featureargs.orb_nfeatures = parser.get<int>( "orbn" );
    if (parser.has("orbscale")) featureargs.orb_scaleFactor = parser.get<float>( "orbscale" );
    if (parser.has("orbnlevels")) featureargs.orb_nlevels = parser.get<int>( "orbnlevels" );
    if (parser.has("orbedge")) featureargs.orb_edgeThreshold = parser.get<int>( "orbedge" );
    if (parser.has("orbfirst")) featureargs.orb_firstLevel = parser.get<int>( "orbfirst" );
    if (parser.has("orbWTA")) featureargs.orb_WTA_K = parser.get<int>( "orbWTA" );
    if (parser.has("orbpatch")) featureargs.orb_patchSize = parser.get<int>( "orbpatch" );
    if (parser.has("orbfast")) featureargs.orb_fastThreshold = parser.get<int>( "orbfast" );
    if (parser.has("orbblur")) featureargs.orb_blurForDescriptor = true;
    if (parser.has("ratio")) featureargs.ratio = parser.get<float>( "ratio" );
    if (parser.has("homo")) featureargs.homo = parser.get<int>( "homo");
    if (parser.has("ransac")) featureargs.ransac = parser.get<double> ("ransac");
    if (parser.has("surfhess")) featureargs.surf_hessianThreshold = parser.get<double>( "surfhess");
    if (parser.has("surfoct")) featureargs.surf_nOctaves = parser.get<int>( "surfoct");
    if (parser.has("surfoctL")) featureargs.surf_nOctaveLayers = parser.get<int>( "surfoctL");
    if (parser.has("surfext")) featureargs.surf_extended = true;
    if (parser.has("surfkey")) featureargs.surf_keypointsRatio = parser.get<float>( "surfkey");
    if (parser.has("surfup")) featureargs.surf_upright = true;
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
	pass_fail = two_file(frame0_name, frame1_name, file, crop_width, scale, top, bottom, features, use_template, args, featureargs);
	  }
    else if (style == 1)
      {
	pass_fail = from_file(frame0_name, file, scale, top, bottom, features,use_template, args, featureargs);
      }
    else if (style == 2)
      {
	pass_fail = average_flow(frame0_name, file, scale, border, args);
      }
    return pass_fail;
}

 int two_file(std::string frame0_name, std::string frame1_name, std::string file, int crop_width, float scale, int top, int bottom, bool features, bool use_template, const OptflowArgs& args, const FeatureArgs& featureargs)
{


    cv::Mat orig_frame0 = imread(frame0_name, cv::IMREAD_GRAYSCALE);
    cv::Mat orig_frame1 = imread(frame1_name, cv::IMREAD_GRAYSCALE);
    cv::Mat frame0, frame1;
    if (frame1.size() != frame0.size())
      {
        std::cerr << "Images should be of equal sizes" << std::endl;
        return -1;
      }

    cv::Rect roi_0;
    cv::Rect roi_1;
    
    roi_0.y = 0;
    roi_1.y = 0;
    roi_0.height = orig_frame0.rows;
    roi_1.height = orig_frame1.rows;
    if (crop_width)
      {
	roi_0.x = orig_frame0.cols - crop_width;
	roi_1.x = 0;
	roi_0.width = crop_width;
	roi_1.width = crop_width;
      }
    else
      {
	roi_0.x = 0;
	roi_1.x = 0;
	roi_0.width = orig_frame0.cols;
	roi_1.width = orig_frame1.cols;
      }
    if (scale != 1)
      {
	cv::resize(orig_frame0(roi_0), frame0, cv::Size(), scale, scale, CV_INTER_LANCZOS4);
	cv::resize(orig_frame1(roi_1), frame1, cv::Size(), scale, scale, CV_INTER_LANCZOS4);
      }
    else
      {
	orig_frame0.copyTo(frame0);
	orig_frame1.copyTo(frame1);
      }

    cv::Rect roi_top_0, roi_top_1, roi_bottom_0, roi_bottom_1;
    std::vector< cv::Rect > rois;
    if (frame0.cols == frame1.cols)
	{
	  roi_top_0.x = 0;
	  roi_bottom_0.x = 0;
	  roi_top_0.width = frame0.cols;
	  roi_bottom_0.width = frame0.cols;
	  roi_top_0.height = 0; 
	  roi_top_1.x = 0;
	  roi_bottom_1.x = 0;
	  roi_top_1.width = frame0.cols;
	  roi_bottom_1.width = frame0.cols;
	  roi_top_1.height = 0; 
	}
      else if (frame0.cols > frame1.cols)
	{
	  roi_top_0.x = (frame0.cols-frame1.cols)/2;
	  roi_top_0.width = frame1.cols;
	  roi_top_1.x = 0;
	  roi_top_1.width = frame1.cols;
	  roi_bottom_0.x = (frame0.cols-frame1.cols)/2;
	  roi_bottom_0.width = frame1.cols;
	  roi_bottom_1.x = 0;
	  roi_bottom_1.width = frame1.cols;
	}
      else
	{
	  roi_top_1.x = (frame1.cols-frame0.cols)/2;
	  roi_top_1.width = frame0.cols;
	  roi_top_0.x = 0;
	  roi_top_0.width = frame0.cols;
	  roi_bottom_1.x = (frame1.cols-frame0.cols)/2;
	  roi_bottom_1.width = frame0.cols;
	  roi_bottom_0.x = 0;
	  roi_bottom_0.width = frame0.cols;
	}
    
    top = top*scale;
    bottom = bottom*scale;
    std::string output_dir = "";
    std::string out_name = file;
          if (!top)
	{
	  roi_top_0.y = 0;
	  roi_top_0.height = frame0.rows;
	  roi_top_1.y = 0;
	  roi_top_1.height = frame1.rows;
	  rois.push_back(roi_top_0);
	  rois.push_back(roi_top_1);
	  
	}
      else
	{
	  roi_top_0.y = 0;
	  roi_top_0.height = top;
	  roi_top_1.y = 0;
	  roi_top_1.height = top;
	  rois.push_back(roi_top_0);
	  rois.push_back(roi_top_1);
	}
      if (bottom)
	{
	  roi_bottom_0.y = frame0.rows-bottom;
	  roi_bottom_0.height= bottom;
	  roi_bottom_1.y = frame0.rows-bottom;
	  roi_bottom_1.height= bottom;
	  rois.push_back(roi_bottom_0);
	  rois.push_back(roi_bottom_1);
	}
    solve_rois(frame0, frame1, output_dir, out_name, rois, features, use_template, args, featureargs);
    
    return 0;
}

 int from_file(std::string file_name, std::string output_dir, float scale, int top, int bottom, bool features, bool use_template, const OptflowArgs& args, const FeatureArgs& featureargs)
{
  std::ifstream infile(file_name.c_str());
  std::string frame0_name, frame1_name, out_name, old_frame0="", old_frame1="";
  cv::Mat frame0, frame1;
  bool temp_features;
  char buffer[200];
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
      cv::Rect roi_top_0, roi_top_1, roi_bottom_0, roi_bottom_1;
      std::vector< cv::Rect > rois;
      if (frame0.cols == frame1.cols)
	{
	  roi_top_0.x = 0;
	  roi_bottom_0.x = 0;
	  roi_top_0.width = frame0.cols;
	  roi_bottom_0.width = frame0.cols;
	  roi_top_0.height = 0; 
	  roi_top_1.x = 0;
	  roi_bottom_1.x = 0;
	  roi_top_1.width = frame0.cols;
	  roi_bottom_1.width = frame0.cols;
	  roi_top_1.height = 0;
	  temp_features = features;
	}
      else if (frame0.cols > frame1.cols)
	{
	  roi_top_0.x = (frame0.cols-frame1.cols)/2;
	  roi_top_0.width = frame1.cols;
	  roi_top_1.x =  0;
	  roi_top_1.width = frame1.cols;
	  temp_features = true;
	}
      else
	{
	  roi_top_1.x = (frame1.cols-frame0.cols)/2;
	  roi_top_1.width = frame0.cols;
	  roi_top_0.x = 0;
	  roi_top_0.width = frame0.cols;
	  temp_features = true;
	}
      if ( frame0.rows > frame1.rows )
	{
	  roi_top_0.y = (frame0.rows-frame1.rows)/2;
	  roi_top_0.width = frame1.rows;
	  roi_top_1.y = 0;
	  roi_top_1.width = frame1.rows;
	}
      else if ( frame1.rows > frame0.rows )
	{
	  roi_top_1.y = (frame1.rows-frame0.rows)/2;
	  roi_top_1.width = frame0.rows;
	  roi_top_0.y = 0;
	  roi_top_0.width = frame0.rows;
	}
      if (!top || frame0.cols != frame1.cols)
	{
	  roi_top_0.y = 0;
	  roi_top_0.height = frame0.rows;
	  roi_top_1.y = 0;
	  roi_top_1.height = frame1.rows;
	  rois.push_back(roi_top_0);
	  rois.push_back(roi_top_1);
	}
      else if ( !top && frame0.rows != frame1.rows)
	{
	  roi_top_0.y = 0;
	  roi_top_0.height = frame0.rows;
	  roi_top_1.y = 0;
	  roi_top_1.height = frame1.rows;
	  rois.push_back(roi_top_0);
	  rois.push_back(roi_top_1);
	}
      else
	{
	  roi_top_0.y = 0;
	  roi_top_0.height = top;
	  roi_top_1.y = 0;
	  roi_top_1.height = top;
	  rois.push_back(roi_top_0);
	  rois.push_back(roi_top_1);
	}
      if (bottom && frame0.cols == frame1.cols && frame0.rows == frame1.rows)
	{
	  roi_bottom_0.y = frame0.rows-bottom;
	  roi_bottom_0.height= bottom;
	  roi_bottom_1.y = frame0.rows-bottom;
	  roi_bottom_1.height= bottom;
	  rois.push_back(roi_bottom_0);
	  rois.push_back(roi_bottom_1);
	}
      std::sprintf(buffer, "%0.2f", scale);

      solve_rois(frame0, frame1, output_dir, out_name+"_"+buffer, rois, temp_features, use_template, args, featureargs);
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


 void solve_rois(cv::Mat frame0, cv::Mat frame1, std::string output_dir, std::string out_name, std::vector<cv::Rect> rois, bool features, bool use_template, const OptflowArgs& args, const FeatureArgs& featureargs)
{
  cv::cuda::GpuMat frame0_GPU, frame1_GPU;
  frame0_GPU.upload(frame0);
  frame1_GPU.upload(frame1);
  cv::cuda::GpuMat flow_GPU;
  cv::cuda::GpuMat old_frame0 = frame0_GPU;
  cv::Mat affine(cv::Size(3,2), CV_32FC1);
  double offset_x, offset_y;
  if (use_template)
    {
      cv::cuda::GpuMat new_frame0;
      cv::cuda::GpuMat min_max;
      cv::Ptr<cv::cuda::TemplateMatching> matcher = cv::cuda::createTemplateMatching(CV_8U, args.temp_method, cv::Size(0,0));
      cv::Rect match_region(frame0_GPU.cols-50,frame0_GPU.rows/2,50,50);
      cv::cuda::GpuMat im_template(frame0_GPU(match_region));
      matcher -> match(frame1_GPU, im_template, min_max);
      double minval, maxval;
      cv::Point minlocation, maxlocation;
      cv::cuda::minMaxLoc(min_max, &minval, &maxval, &minlocation, &maxlocation);
      cv::Point2f matchlocation;
      if( args.temp_method  == CV_TM_SQDIFF || args.temp_method == CV_TM_SQDIFF_NORMED )
	{ matchlocation = minlocation; }
      else
	{ matchlocation = maxlocation; }
      std::cout << matchlocation << "\n";
      affine.at<float>(0,0) = 1;
      affine.at<float>(0,1) = 0;
      affine.at<float>(0,2) = matchlocation.x - frame0_GPU.cols + 50;
      affine.at<float>(1,0) = 0;
      affine.at<float>(1,1) = 1;
      affine.at<float>(1,2) = matchlocation.y - frame0_GPU.rows/2;
      cv::cuda::warpAffine(frame0_GPU, new_frame0, affine, frame0_GPU.size(), cv::INTER_LINEAR);

    }
  else if (features)
    {

      cv::cuda::GpuMat new_frame0;
      find_alignment(frame0_GPU(rois.at(0)), frame1_GPU(rois.at(1)), affine, featureargs);
      cv::cuda::warpAffine(frame0_GPU, new_frame0, affine, frame0_GPU.size(), cv::INTER_LINEAR);
      frame0_GPU = new_frame0;
    }
  if ( rois.size() == 2 )
    {
      solve_wrapper(frame0_GPU(rois.at(0)), frame1_GPU(rois.at(1)), output_dir, out_name, features, use_template, affine, args);
    }
  else
    {
      if ( rois.at(0).height > 0)
	{
	  solve_wrapper(frame0_GPU(rois.at(0)), frame1_GPU(rois.at(1)), output_dir, out_name+"_top", features, use_template, affine, args);
	}
      if ( rois.at(1).height > 0)
	{
	  solve_wrapper(frame0_GPU(rois.at(2)), frame1_GPU(rois.at(3)), output_dir, out_name+"_bottom", features, use_template, affine, args);
	}
    }
  if (features || use_template)
    {
      frame0_GPU = old_frame0;
    }
}

 void solve_wrapper(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, std::string output_dir, std::string out_name, bool features, bool use_template, cv::Mat affine, const OptflowArgs& args)
{
  cv::cuda::GpuMat flow_GPU;
  TVL1_solve(frame0, frame1, flow_GPU, args);
  cv::cuda::GpuMat inv_x_GPU, inv_y_GPU;

  if (features || use_template)
    {
      cv::cuda::buildWarpAffineMaps(affine, true, frame0.size(), inv_x_GPU, inv_y_GPU); //inverse
    }
  
  std::string file_x = output_dir+"/"+out_name+"_x.tiff";
  std::string file_y = output_dir+"/"+out_name+"_y.tiff";
  
  std::vector<cv::cuda::GpuMat> flow_xy_GPU;
  cv::cuda::split(flow_GPU,flow_xy_GPU);
  cv::Mat flow_x, flow_y;
  flow_xy_GPU[0].download(flow_x);
  flow_xy_GPU[1].download(flow_y);
  if (features || use_template)
    {
      cv::Mat inv_x, inv_y;
      inv_x_GPU.download(inv_x);
      inv_y_GPU.download(inv_y);
      cv::Mat map_x, map_y;
      map_x.create(inv_x.size(), CV_32FC1);
      map_y.create(inv_y.size(), CV_32FC1);
      for(int j=0; j<inv_x.rows; j++)
	{
	  for ( int i=0; i < inv_x.cols; i++)
	    {
	      map_x.at<float>(j,i) = (float)i;
	      map_y.at<float>(j,i) = (float)j;
	    }
	}
      flow_x += inv_x - map_x;
      flow_y += inv_y - map_y;
      cv::Mat frame0_CPU;
      cv::cuda::GpuMat frame0_remap;
      cv::cuda::remap(frame0, frame0_remap, inv_x_GPU, inv_y_GPU, cv::INTER_LINEAR);
      frame0_remap.download(frame0_CPU);
      frame0_CPU.convertTo(frame0_CPU,CV_32FC1);
      
      for(int j=0; j<frame0_CPU.rows; j++)
	{
	  for( int i=0; i < frame0_CPU.cols; i++)
	    {
	      if (frame0_CPU.at<float>(j,i) == (float)0)
		{
		  flow_x.at<float>(j,i) = (float)0;
		  flow_y.at<float>(j,i) = (float)0;
		}
	    }
	}
      
    }
  imwrite(file_x, flow_x);
  imwrite(file_y, flow_y);
}

void TVL1_solve(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::cuda::GpuMat& output, const OptflowArgs& args)
{
  cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> solver = cv::cuda::OpticalFlowDual_TVL1::create(args.tau, args.lambda, args.theta, args.nscales, args.warps, args.epsilon, args.iterations, args.scaleStep, args.gamma);
  solver -> calc(frame0, frame1, output);
}
