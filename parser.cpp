#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>


void parse_parameters(cv::CommandLineParser& parser, bool features&, bool use_template&, const OptflowArgs& args, const FeatureArgs& featureargs)
{
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
}
