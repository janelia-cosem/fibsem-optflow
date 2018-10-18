#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>

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

void parse_parameters(cv::CommandLineParser& parser, bool features&, bool use_template&, const OptflowArgs& args, const FeatureArgs& featureargs);

#endif
