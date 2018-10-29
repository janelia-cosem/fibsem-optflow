#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <vector>
#include <algorithm> 

#include <curl/curl.h>
#include <jsoncpp/json/json.h>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

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

int main(int argc, const char* argv[])
{
  const std::string keys = "{@filename | | json file}" "{help h | | show help message}";
  Json::Value args;
  Json::Reader reader;
  
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.has("help"))
      {
        parser.printMessage();
        return 0;
      }
  std::string filename = parser.get<std::string>( "@filename" );
  //Decompress if it's a gunzipped file
  if ( (filename.size() >= 3) && (filename.substr(filename.size() - 3) == ".gz"))
    {
      std::ifstream file(filename,std::ios_base::in | std::ios_base::binary);
      std::stringstream json_str;
      boost::iostreams::filtering_streambuf< boost::iostreams::input >  in;
      in.push(boost::iostreams::gzip_decompressor());
      in.push(file);
      boost::iostreams::copy(in, json_str);
      reader.parse(json_str, args, false);
    }
  else
    {
      std::ifstream file(filename, std::ifstream::binary);
      reader.parse(file, args, false);

    }

    int pass_fail;

    int style = args.get("style", 1).asInt();
    if (style == 1)
      {
	pass_fail = from_file(args);
      }
    // else if (style == 2)
    //   {
    // 	pass_fail = average_flow(args);
    //   }
    return pass_fail;
}


int from_file(Json::Value& args)
{
  
  std::string frame0_name, frame1_name, out_name, old_frame0="", old_frame1="";
  cv::Mat frame0, frame1, temp_frame;
  float scale, old_scale;
  char buffer[10]; //Even 10 is overkill
  Json::Value images=args["images"];
  
  for (Json::Value::ArrayIndex i=0; i != images.size(); i++)
    {
      Json::Value im_data=images[i];
      frame0_name = im_data["p"].asString();
      frame1_name = im_data["q"].asString();
      scale = im_data.get("scale", args.get("scale", 0.5).asFloat()).asFloat();
      im_data["scale"] = im_data.get("scale",scale).asDouble();

      //Check to see if one of these images is already in memory.
      //GPU upload is typically small since we use only a fraction.
      if ( (frame0_name == old_frame1) && (scale == old_scale) )
	{
	  temp_frame = frame0;
	  frame0 = frame1; //Already been scaled
	  frame1 = temp_frame;
	  temp_frame.release();
	}
      if ( (frame0_name != old_frame0) || (scale != old_scale) ) //If it is equal it's already assigned
	{
	  frame0 = cv::imread(frame0_name, cv::IMREAD_GRAYSCALE);
	  if (scale != 1) cv::resize(frame0, frame0, cv::Size(), scale, scale);
	}
      if ( (frame1_name != old_frame1) || (scale != old_scale) )
	{
	  if ((frame1_name != old_frame0) || (scale != old_scale) )
	    {
	      frame1 = cv::imread(frame1_name, cv::IMREAD_GRAYSCALE);
	      if (scale != 1) cv::resize(frame1, frame1, cv::Size(), scale, scale);
	    }
	}
      old_scale = scale;
      old_frame0 = frame0_name;
      old_frame1 = frame1_name;

      Json::Value rois;
      if (im_data.isMember("rois"))
	{
	  //We have specific instructions
	  get_rois(rois,images["rois"], std::min(frame0.rows, frame1.rows), std::min(frame0.cols, frame1.cols));
	}
      else if (args.isMember("rois"))
	{
	  //Use base values
	  get_rois(rois,args["rois"], std::min(frame0.rows, frame1.rows), std::min(frame0.cols, frame1.cols));
	}
      else
	{
	  //Just set to be min sizes and then features will fix it.
	  rois["default"][0] = 0;
	  rois["default"][1] = 0;
	  rois["default"][2] = std::min(frame0.cols, frame1.cols);
	  rois["default"][3] =  std::min(frame0.rows, frame1.rows);
	}
      std::sprintf(buffer, "%0.2f", scale);

      im_data["output"] = im_data.get("output", args["output_dir"].asString()+"/"+im_data["output_name"].asString()+"_"+buffer);
      solve_rois(frame0, frame1, rois, im_data, args);
    }
  return 0;
}

  /* Kept here in case of implementation in future with a json option*/
// int average_flow(std::string file_name, std::string output_dir, float scale, int border, const OptflowArgs& args)
// {
//   std::string im_name;
//   std::ifstream infile(file_name.c_str());
//   std::deque<cv::Mat> frames;
//   cv::Mat curr_frame;
//   std::vector<std::string> fnames;
//   //Magic numbers (e**-x^2/4 resummed to one)
//   cv::Vec6f weights(exp(-9./4.), exp(-1.), exp(-1./4.), exp(-1./4.), exp(-1.), exp(-9./4.));

//   weights = weights * 0.5/(exp(-9./4.)+exp(-1.)+exp(-1./4.));

//   //Read file in
//   while (getline(infile,im_name))
//     {
//       fnames.push_back(im_name);
//     }
  
//   //Won't be too large
//   int n_files = static_cast<int>(fnames.size());
//   curr_frame = imread(fnames.at(0), cv::IMREAD_GRAYSCALE);
  
//   cv::Mat blur_image(curr_frame.size(), CV_64F, cv::Scalar(0));
//   cv::Mat blur_8;
//   //Initialise matricies;
//   frames.push_back(blur_image); //Will be popped
//   frames.push_back(curr_frame);
//   curr_frame.release(); //It's done its job
//   frames.push_back(imread(fnames.at(1), cv::IMREAD_GRAYSCALE));
//   frames.push_back(imread(fnames.at(2), cv::IMREAD_GRAYSCALE));
//   frames.push_back(imread(fnames.at(3), cv::IMREAD_GRAYSCALE));
//   frames.push_back(imread(fnames.at(4), cv::IMREAD_GRAYSCALE));
//   frames.push_back(imread(fnames.at(5), cv::IMREAD_GRAYSCALE));
  
//   for (int i=3; i<n_files-3; i++)
//     {
//       std::cout << "N: " << std::to_string(i) << " " << fnames.at(i) << "\n";
//       frames.pop_front();
//       frames.push_back(imread(fnames.at(i+3), cv::IMREAD_GRAYSCALE));
//       blur_image = weights(0)*frames.at(0) + weights(1)*frames.at(1) + weights(2)*frames.at(2) + weights(3)*frames.at(4) + weights(4)*frames.at(5) + weights(5)*frames.at(6);
//       blur_image.convertTo(blur_8, CV_8U, 1.);
//       remap_and_save(output_dir, i, frames.at(3), blur_8, scale, border, args);
//     }
      
//   return 0;
// }

void get_rois(Json::Value& rois, Json::Value& args, int rows, int cols)
{
  if ( args.isMember("top") )
    {
      rois["top"][0] = 0;
      rois["top"][1] = 0;
      rois["top"][2] = cols;
      rois["top"][3] = args.get("top",300).asInt();
    }
  if ( args.isMember("bottom") )
    {
      int bottom=args.get("bottom", 300).asInt();
      rois["bottom"][0] = 0;
      rois["bottom"][1] = rows-bottom;
      rois["bottom"][2] = cols;
      rois["bottom"][3] = bottom;
    }
  if ( args.isMember("custom") )
    {
      if ( args["custom"].isMember("0") ) //Has potentially different ones
	{
	  rois["custom_diff"]["0"] = args["custom"]["0"];
	  if ( !args["custom"].isMember("1") )
	    {
	      std::cerr << "If you specify a custom for the first frame, you must specify a custom for the second. Seg fault on it's way.\n";
	    }
	  rois["custom_diff"]["1"] = args["custom"]["1"];
	}
      else
	{
	  rois["custom"] = args["custom"];
	}
    }
}

/*To implement again in future, but not immediately needed for JSON.*/
// void remap_and_save(std::string output_dir, int i, cv::Mat frame, cv::Mat blur, float scale, int border, const OptflowArgs& args)
// {
//   cv::cuda::GpuMat frame_GPU, blur_GPU, flow_GPU;
//   cv::Mat scale_flow, flow;
//   cv::Mat scale_frame;
//   resize(frame, scale_frame, cv::Size(), scale, scale);
//   resize(blur, blur, cv::Size(), scale, scale);
//   frame_GPU.upload(scale_frame);
//   blur_GPU.upload(blur);
//   TVL1_solve(frame_GPU, blur_GPU, flow_GPU, args);
//   flow_GPU.download(scale_flow);
//   scale_flow = 1/scale * scale_flow;
//   resize(scale_flow, flow, cv::Size(), 1/scale, 1/scale);

//   cv::Mat new_frame(frame.rows+2*border, frame.cols+2*border, CV_8U);
//   cv::Mat framed_frame;
//   cv::Mat framed_flow;
//   cv::copyMakeBorder(frame, framed_frame, border, border, border, border, 0);
  
//   cv::copyMakeBorder(flow, framed_flow, border, border, border, border, 0);

//   cv::Mat map(flow.size(), CV_32FC2);
//   for (int y = 0; y < map.rows; y++)
//     {
//       for (int x = 0; x < map.cols; x++)
// 	{
// 	  cv::Point2f df = flow.at<cv::Point2f>(y,x);
// 	  map.at<cv::Point2f>(y,x) = cv::Point2f(x-df.x, y-df.y);
// 	}
//     }
//   cv::Mat framed_map;
//   cv::copyMakeBorder(map, framed_map,  border, border, border, border, 0);
//   std::vector<cv::Mat> map_xy;
//   cv::split(framed_map, map_xy);
//   cv::remap(framed_frame, new_frame, map_xy[0], map_xy[1], 1);
//   cv::imwrite(output_dir+"/"+std::to_string(i)+".tiff", new_frame);
// }

cv::Rect roi_from_array(Json::Value& roi_array)
{
  cv::Rect roi;
  roi.x = roi_array[0u].asInt(); //0u is safer
  roi.y = roi_array[1].asInt();
  roi.width = roi_array[2].asInt();
  roi.height = roi_array[3].asInt();
  return roi;
}

void solve_rois(cv::Mat& frame0, cv::Mat& frame1, Json::Value& rois, Json::Value& im_args, Json::Value& args)
{
  cv::cuda::GpuMat frame0_GPU, frame1_GPU;
  frame0_GPU.upload(frame0);
  frame1_GPU.upload(frame1);
  cv::cuda::GpuMat flow_GPU;
  cv::cuda::GpuMat old_frame0 = frame0_GPU;
  cv::Mat affine(cv::Size(3,2), CV_32FC1);
  double offset_x, offset_y;
  bool features;
  std::vector < cv::Rect > roi_vec;

  if ( im_args.isMember("features") && !im_args["features"].asBool() )
    {
      features = false;
    }
  else if ( args.isMember("features") && !args["features"].asBool() )
    {
      features = false;
    }
  else if ( im_args.get("features",false).asBool() || args.get("features",false).asBool() )
    {
      features = true;
    }
  else
    {
      features = false;
    }
  for (auto const& roi_key: rois.getMemberNames())
    {
      if ( (roi_key == "top") || (roi_key == "bottom") )
	{
	  im_args["output_suffix"] = "_"+roi_key;
	}
      else
	{
	  im_args["output_suffix"] = "";
	}
      if (roi_key == "custom_diff")
	{
	  if (features)
	    {
	      std::cerr << "Features isn't compatible with different ROIs for each image.\n Ignoring features.\n";
	    }
	  cv::Rect roi_0, roi_1;
	  roi_0 = roi_from_array(rois["custom_diff"]["0"]);
	  roi_1 = roi_from_array(rois["custom_diff"]["1"]);
	  roi_vec.push_back(roi_0);
	  roi_vec.push_back(roi_1);
	  solve_wrapper(frame0_GPU(roi_0), frame1_GPU(roi_1), affine, im_args, args, features, roi_vec);
	}
      else
	{
	  if (features || (frame0.rows != frame1.rows) || (frame0.cols != frame1.cols) || roi_key=="default")
	    {
	      if ( ((frame0.rows != frame1.rows) || (frame0.cols != frame1.cols)) || roi_key=="default" && roi_key != "custom" && features == false)
		{
		  std::cerr << "Rows or columns differ between frames no ROI selected, reverting to features even though it wasn't selected.\n";
		}
	      cv::cuda::GpuMat new_frame1;
	      find_alignment(frame1_GPU, frame0_GPU, affine, im_args, args);
	      cv::cuda::warpAffine(frame1_GPU, new_frame1, affine, frame0_GPU.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
	      frame1_GPU = new_frame1;
	      features = true;
	    }
	  cv::Rect roi;
	  roi = roi_from_array(rois[roi_key]);
	  roi_vec.push_back(roi);
	  roi_vec.push_back(roi);
	  solve_wrapper(frame0_GPU(roi), frame1_GPU(roi), affine, im_args, args, features, roi_vec);
	}

    }

  if (im_args.get("output_type",args.get("output_type","map").asString()).asString() == "random_points")
    {
      upload_points(im_args, args);
    }

}

/*GpuMats are ROI subsets so can't be constant*/
void solve_wrapper(cv::cuda::GpuMat frame0, cv::cuda::GpuMat frame1, cv::Mat affine, Json::Value& im_args, Json::Value& args, bool features, std::vector < cv::Rect > roi_vec)
{
  cv::cuda::GpuMat flow_GPU;
  Json::Value TV_args;
  TV_args = generate_TV_args(im_args,args);
  TVL1_solve(frame0, frame1, flow_GPU, TV_args);
  
  
  std::vector<cv::cuda::GpuMat> flow_xy_GPU;
  cv::cuda::split(flow_GPU,flow_xy_GPU);
  cv::Mat flow_x, flow_y;
  cv::cuda::GpuMat warp_frame0;
  std::string output_type;
  bool debug=args.get("debug",false).asBool();
  output_type = im_args.get("output_type",args.get("output_type","map").asString()).asString();

  if (features)
    {
      cv::Mat map_x, map_y;
      cv::cuda::GpuMat map_x_GPU, map_y_GPU;
      cv::cuda::GpuMat new_x_GPU, new_y_GPU;

      map_x.create(flow_xy_GPU[0].size(), CV_32FC1);
      map_y.create(flow_xy_GPU[1].size(), CV_32FC1);
      for(int j=0; j < map_x.rows; j++)
	{
	  for ( int i=0; i < map_x.cols; i++)
	    {
	      map_x.at<float>(j,i) = (float)i;
	      map_y.at<float>(j,i) = (float)j;
	    }
	}
      map_x_GPU.upload(map_x);
      map_y_GPU.upload(map_y);
      cv::cuda::add(flow_xy_GPU[0], map_x_GPU, flow_xy_GPU[0]);
      cv::cuda::add(flow_xy_GPU[1], map_y_GPU, flow_xy_GPU[1]);
      cv::cuda::warpAffine(flow_xy_GPU[0], new_x_GPU, affine, flow_xy_GPU[0].size(), cv::INTER_LINEAR+cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, 0);
      cv::cuda::warpAffine(flow_xy_GPU[1], new_y_GPU, affine, flow_xy_GPU[1].size(), cv::INTER_LINEAR+cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, 0);

      if (output_type == "flow")
	{
	  cv::cuda::subtract(new_x_GPU, map_x_GPU, flow_xy_GPU[0]);
	  cv::cuda::subtract(new_y_GPU, map_y_GPU, flow_xy_GPU[1]);
	}
      else
	{
	  flow_xy_GPU[0] = new_x_GPU;
	  flow_xy_GPU[1] = new_y_GPU;
	}
    }
  else if (output_type == "map")
    {
      cv::Mat map_x, map_y;
      cv::cuda::GpuMat map_x_GPU, map_y_GPU;
      cv::cuda::GpuMat new_x_GPU, new_y_GPU;
      
      map_x.create(flow_xy_GPU[0].size(), CV_32FC1);
      map_y.create(flow_xy_GPU[1].size(), CV_32FC1);
      for(int j=0; j < map_x.rows; j++)
	{
	  for ( int i=0; i < map_x.cols; i++)
	    {
	      map_x.at<float>(j,i) = (float)i;
	      map_y.at<float>(j,i) = (float)j;
	    }
	}
      map_x_GPU.upload(map_x);
      map_y_GPU.upload(map_y);
      
      cv::cuda::add(flow_xy_GPU[0], map_x_GPU, flow_xy_GPU[0]);
      cv::cuda::add(flow_xy_GPU[1], map_y_GPU, flow_xy_GPU[1]);
    }
  //Mask out 0s in frame0, these shouldn't actually map to anything so if something has happened it's wrong
  cv::cuda::GpuMat mask;
  cv::cuda::threshold(frame1, mask, 1.0, 1.0, cv::THRESH_BINARY_INV);
  flow_xy_GPU[0].setTo(cv::Scalar::all(0), mask);
  flow_xy_GPU[1].setTo(cv::Scalar::all(0), mask);

  flow_xy_GPU[0].download(flow_x);
  flow_xy_GPU[1].download(flow_y);


  if ( (output_type == "map") || (output_type == "flow") )
    {
      std::string file_x = im_args["output"].asString()+im_args["output_suffix"].asString()+"_x.tiff";
      std::string file_y = im_args["output"].asString()+im_args["output_suffix"].asString()+"_y.tiff";
      cv::imwrite(file_x, flow_x);
      cv::imwrite(file_y, flow_y);
    }

  if ( (output_type == "random_points" ))
    {
      random_points(flow_x, flow_y, im_args, args, roi_vec);
    }
}
  


Json::Value generate_TV_args(const Json::Value& im_args,const Json::Value& args)
{
  Json::Value TV_args;
  TV_args["tau"] = im_args.get("tau",args.get("tau",0.25).asDouble()).asDouble();
  TV_args["lambda"] = im_args.get("lambda",args.get("lambda",0.05).asDouble()).asDouble();
  TV_args["theta"] = im_args.get("theta",args.get("theta",0.3).asDouble()).asDouble();
  TV_args["nscales"] = im_args.get("nscales",args.get("nscales",10).asInt()).asInt();
  TV_args["warps"] = im_args.get("warps",args.get("warps",5).asInt()).asInt();
  TV_args["epsilon"] = im_args.get("epsilon",args.get("epsilon",0.01).asDouble()).asDouble();
  TV_args["iterations"] = im_args.get("iterations",args.get("iterations",300).asInt()).asInt();
  TV_args["scaleStep"] = im_args.get("scaleStep",args.get("scaleStep",0.8).asDouble()).asDouble();
  TV_args["gamma"] = im_args.get("gamma",args.get("gamma",0.0).asDouble()).asDouble();
  TV_args["useInitialFlow"] = im_args.get("useInitialFlow",args.get("useInitialFlow",false).asBool()).asBool();
  return TV_args;
}

void TVL1_solve(cv::cuda::GpuMat& frame0, cv::cuda::GpuMat& frame1, cv::cuda::GpuMat& output, const Json::Value& args)
{
  cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> solver = cv::cuda::OpticalFlowDual_TVL1::create(args["tau"].asDouble(), args["lambda"].asDouble(), args["theta"].asDouble(), args["nscales"].asInt(), args["warps"].asInt(), args["epsilon"].asDouble(), args["iterations"].asInt(), args["scaleStep"].asDouble(), args["gamma"].asDouble());
  solver -> calc(frame0, frame1, output);
}

void random_points(cv::Mat& flow_x, cv::Mat& flow_y, Json::Value& im_args, const Json::Value& args, std::vector < cv::Rect > roi_vec)
{
  Json::Value pm;
  
  Json::Value p,q;
  bool debug=args.get("debug",false).asBool();
  float scale = im_args.get("scale", args.get("scale", 0.5).asFloat()).asFloat();
  float inv_scale = 1./scale;
  cv::RNG rng(cv::getTickCount());


  if (debug)
    {
      cv::RNG rng(); //Start with same random each time, doubling up to over-ride stupid compilation errors
    }
  int count = 0;
  cv::Point pos;
  while (count < im_args.get("npoints",args.get("npoints",25).asInt()).asInt())
    {
      pos.x = rng.uniform(0,flow_x.cols);
      pos.y = rng.uniform(0,flow_x.rows);
      
      if (flow_x.at<float>(pos.y,pos.x) != 0) //0s are masked out
	{
	  im_args["point_matches"]["w"].append(1); //Because of course
	  
	  p[0] = (pos.x + roi_vec.at(0).x) * inv_scale;
	  p[1] = (pos.y + roi_vec.at(0).y) * inv_scale;
	  
	  q[0] = (pos.x + roi_vec.at(1).x + flow_x.at<float>(pos.y,pos.x)) * inv_scale;
	  q[1] = (pos.y + roi_vec.at(1).y + flow_y.at<float>(pos.y,pos.x)) * inv_scale;
	  im_args["point_matches"]["p"].append(p);
	  im_args["point_matches"]["q"].append(q);
	  count += 1;
	}
    }
}


void upload_points(const Json::Value& im_args, const Json::Value& args)
{
  bool debug=args.get("debug",false).asBool();

  Json::StreamWriterBuilder builder;
  builder["commentStyle"] = "None";
  builder["indentation"] = "   ";
  Json::Value payload;
  Json::Value single_pair;
  single_pair["pGroupId"] = im_args["pGroupId"];
  single_pair["pId"] = im_args["pId"];
  single_pair["qGroupId"] = im_args["qGroupId"];
  single_pair["qId"] = im_args["qId"];
  single_pair["matches"] = im_args["point_matches"];
  payload[0] = single_pair;
  std::string owner = args.get("owner","flyem").asString();
  std::string matchCollection = args.get("matchCollection","forgetful_owner").asString();
  std::string host = args.get("host", "10.40.3.162").asString();
  std::string port = args.get("port", "8080").asString();
  std::string hostname;
  CURL *curl;
  CURLcode res;

  curl_global_init(CURL_GLOBAL_ALL);
  std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
  std::string payload_str = Json::writeString(builder, payload);
  curl = curl_easy_init();
  if (curl) {
    hostname = "http://"+host+":"+port+"/render-ws/v1/owner/"+owner+"/matchCollection/"+matchCollection+"/matches";
    curl_easy_setopt(curl, CURLOPT_URL, hostname.c_str());
    if (debug)
      {
	std::cout << hostname;
      }

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
    curl_easy_setopt(curl, CURLOPT_COPYPOSTFIELDS, payload_str.c_str());
    res = curl_easy_perform(curl);
    if (res != CURLE_OK)
      {
	std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << "\n" <<"Hostname: " << hostname << "\n";
      }
    curl_easy_cleanup(curl);
    }
}
