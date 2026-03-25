#pragma once

#include <vector>

#include "matching_algorithm.h"

namespace scan_lock {

struct GICPParams {
  std::vector<float> resolutions{0.4f, 0.2f};
  int max_iterations{100};
  float correspondence_distance_factor{4.0f};
};

class GICP : public MatchingAlgorithm {
public:
  explicit GICP(const GICPParams& params = GICPParams{});

  bool align(const PointCloud::ConstPtr& source,
             const PointCloud::ConstPtr& target,
             const Eigen::Matrix4f& initial_guess,
             Eigen::Matrix4f& result) override;

private:
  GICPParams params_;
};

}  // namespace scan_lock
