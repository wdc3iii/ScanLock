#pragma once

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace scan_lock {

using PointType = pcl::PointXYZINormal;
using PointCloud = pcl::PointCloud<PointType>;

class MatchingAlgorithm {
public:
  virtual ~MatchingAlgorithm() = default;

  /// Align source cloud to target cloud given an initial guess.
  /// Returns true on convergence, populating result with the refined transform.
  virtual bool align(const PointCloud::ConstPtr& source,
                     const PointCloud::ConstPtr& target,
                     const Eigen::Matrix4f& initial_guess,
                     Eigen::Matrix4f& result) = 0;
};

}  // namespace scan_lock
