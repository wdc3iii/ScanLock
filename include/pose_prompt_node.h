#pragma once

#include <memory>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

using PosePromptPointType = pcl::PointXYZI;
using PosePromptPointCloud = pcl::PointCloud<PosePromptPointType>;

/// Standalone RViz-prompting helper for scan_lock. Loads the map PCD
/// independently of scan_lock_node and publishes a coarse-to-fine pair of
/// visualization clouds so an operator can set /initialpose without needing
/// scan_lock_node (or any GPU) to be running:
///   - map_cloud_coarse: the whole map, heavily downsampled, for fast
///     orbiting/panning even under CPU-only rendering.
///   - map_cloud_fine: a small region around the last "Publish Point" click,
///     downsampled much less, for precise final pose placement.
/// The final pose is still set via RViz's "2D Pose Estimate" tool, which
/// publishes directly to /initialpose exactly as before -- this node never
/// touches that topic.
class PosePromptNode : public rclcpp::Node {
public:
  explicit PosePromptNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  void region_point_callback(
      const geometry_msgs::msg::PointStamped::ConstSharedPtr msg);

  PosePromptPointCloud::Ptr map_cloud_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_coarse_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_fine_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_region_point_;

  std::string map_frame_;
  double fine_voxel_size_{0.2};
  double fine_region_radius_{20.0};
  int fine_point_cap_{1000000};
};
