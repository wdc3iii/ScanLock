#include "gicp.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>

namespace scan_lock {

GICP::GICP(const GICPParams& params) : params_(params) {}

bool GICP::align(const PointCloud::ConstPtr& source,
                 const PointCloud::ConstPtr& target,
                 const Eigen::Matrix4f& initial_guess,
                 Eigen::Matrix4f& result) {
  if (!source || source->empty() || !target || target->empty()) {
    return false;
  }

  Eigen::Matrix4f current_guess = initial_guess;

  for (const float res : params_.resolutions) {
    // Voxel filter source
    PointCloud::Ptr filtered_source(new PointCloud());
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(res, res, res);
    voxel.setInputCloud(source);
    voxel.filter(*filtered_source);

    // Voxel filter target
    PointCloud::Ptr filtered_target(new PointCloud());
    voxel.setInputCloud(target);
    voxel.filter(*filtered_target);

    if (filtered_source->empty() || filtered_target->empty()) {
      return false;
    }

    // Run GICP
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
    gicp.setMaxCorrespondenceDistance(params_.correspondence_distance_factor * res);
    gicp.setMaximumIterations(params_.max_iterations);
    gicp.setInputSource(filtered_source);
    gicp.setInputTarget(filtered_target);

    PointCloud aligned;
    gicp.align(aligned, current_guess);

    if (!gicp.hasConverged()) {
      return false;
    }

    current_guess = gicp.getFinalTransformation();
  }

  result = current_guess;
  return true;
}

}  // namespace scan_lock
