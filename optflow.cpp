#include <iostream>
#include <fstream>

#include <opencv2/core/utility.hpp>
#include "opencv2/video.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>

#include <gsl/gsl_qrng.h>

using namespace cv;
using namespace std;
using namespace cv::cuda;

int two_file(string frame0_name, string frame1_name, string file, string algo, int crop_width, int n_pts, float scale)
{

    gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_sobol, 2);

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
    GpuMat frame0_GPU, frame1_GPU, flow_GPU, ptstotrack_GPU;
    GpuMat status_GPU, error_GPU;
    Mat_<Point2f> flow;
    vector<Point2f> ptstotrack(n_pts);

    if (algo == "SparsePLK")
      {
	for (int i=0; i < n_pts; i++)
	  {
	    double v[2];
	    gsl_qrng_get (q, v);
	    ptstotrack[i].x = v[0]*frame0.cols;
	    ptstotrack[i].y = v[1]*frame0.rows;
	  }
	ptstotrack_GPU.upload(ptstotrack);
      }
    gsl_qrng_free (q);
    if (algo == "Brox")
      {
    	frame0.convertTo(frame0, CV_32F, 1.0/255.0);
	frame1.convertTo(frame1, CV_32F, 1.0/255.0);
      }
    
    frame0_GPU.upload(frame0);
    frame1_GPU.upload(frame1);
  
    if (algo == "TVL")
      {
	
	Ptr<OpticalFlowDual_TVL1> solver = cv::cuda::OpticalFlowDual_TVL1::create(0.25, 0.05/scale, 0.2);
	solver -> calc(frame0_GPU, frame1_GPU, flow_GPU);
      }
    else if (algo == "PLK")
      {
	Ptr<DensePyrLKOpticalFlow> solver = cv::cuda::DensePyrLKOpticalFlow::create();
	solver -> calc(frame0_GPU, frame1_GPU, flow_GPU);
      }
    else if (algo == "Brox")
      {
	Ptr<BroxOpticalFlow> solver = cv::cuda::BroxOpticalFlow::create();
	solver -> calc(frame0_GPU, frame1_GPU, flow_GPU);
      }
    else if (algo == "SparsePLK")
      {
	Ptr<cv::cuda::SparsePyrLKOpticalFlow> solver = cv::cuda::SparsePyrLKOpticalFlow::create(Size(51,51)) ;
	solver -> calc(frame0_GPU, frame1_GPU, ptstotrack_GPU, flow_GPU, status_GPU, error_GPU);
      }


    flow_GPU.download(flow);
    
    string file_x = file+ "_x.tiff", file_y = file + "_y.tiff";
    vector<Mat> flow_xy;
    Mat error;
    split(flow, flow_xy);
    if (algo == "SparsePLK")
      {
	Mat x_pts(Size(n_pts,1), CV_64F);
	Mat y_pts(Size(n_pts,1), CV_64F);
	error_GPU.download(error);
	for (int i=0; i < n_pts; i++)
	  {
	    x_pts.at<double>(i) = ptstotrack[i].x;
	    y_pts.at<double>(i) = ptstotrack[i].y;
	  }
	x_pts.convertTo(x_pts, CV_32F);
	y_pts.convertTo(y_pts, CV_32F);
	error.convertTo(error, CV_32F);
	flow_xy[0].push_back(x_pts);
	flow_xy[0].push_back(error);
	flow_xy[1].push_back(y_pts);
	flow_xy[1].push_back(error);
      }
    imwrite(file_x, flow_xy[0]);
    imwrite(file_y, flow_xy[1]);

    
    return 0;
}

int from_file(string file_name, string output_dir, float scale)
{
  ifstream infile(file_name.c_str());
  string frame0_name, frame1_name, out_name;

  while (infile >> frame0_name >> frame1_name >> out_name)
    {
      printf("%s %s\n", frame0_name.c_str(), frame1_name.c_str());
      Mat frame0 = imread(frame0_name, IMREAD_GRAYSCALE);
      Mat frame1 = imread(frame1_name, IMREAD_GRAYSCALE);
      if (scale != 1)
      {
	resize(frame0, frame0, Size(), scale, scale);
	resize(frame1, frame1, Size(), scale, scale);
      }
      GpuMat frame0_GPU, frame1_GPU, flow_GPU;
      Mat_<Point2f> flow;
          
      frame0_GPU.upload(frame0);
      frame1_GPU.upload(frame1);
      Ptr<OpticalFlowDual_TVL1> solver = cv::cuda::OpticalFlowDual_TVL1::create(0.25, 0.05/scale, 0.2);
      solver -> calc(frame0_GPU, frame1_GPU, flow_GPU);
      flow_GPU.download(flow);
      string file_x = output_dir+"/"+out_name+"_x.tiff";
      string file_y = output_dir+"/"+out_name+"_y.tiff";
      vector<Mat> flow_xy;
      split(flow, flow_xy);

      imwrite(file_x, flow_xy[0]);
      imwrite(file_y, flow_xy[1]);
    }

  return 0;
}

  
int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, "{help h || show help message}"
            "{ @output | | output flow}{ @frame0 | | frame 0}{ frame1 | | frame}{ algo | TVL | algorithm}{ crop | 0 | crop size}{ n_pts | 1 | n_pts}{ style | 0 | style}{scale | 0.5 | scale}");
    if (parser.has("help"))
      {
        parser.printMessage();
        return 0;
      }

    string frame0_name = parser.get<string>("@frame0");
    string frame1_name = parser.get<string>("frame1");
    string file = parser.get<string>("@output");
    string algo = parser.get<string>("algo");
    int crop_width = parser.get<int>("crop");
    int n_pts = parser.get<int>("n_pts");
    int style = parser.get<int>("style");
    float scale = parser.get<float>("scale");
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
	pass_fail = two_file(frame0_name, frame1_name, file, algo, crop_width, n_pts, scale);
	  }
    else if (style == 1)
      {
	pass_fail = from_file(frame0_name, file, scale);
      }
    return pass_fail;
}

