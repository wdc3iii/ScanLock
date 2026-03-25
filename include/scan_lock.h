#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "matching_algorithm.h"

namespace scan_lock {

enum class State {
  WAITING_FOR_TF,
  GLOBAL_REGISTRATION,
  LOCALIZED
};

class ScanLockNode : public rclcpp::Node {
public:
  explicit ScanLockNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  // --- Registration methods ---

  /// Perform global registration of scan (odom frame) against map (map frame).
  /// Uses map_T_odom_ as initial guess. Returns true on success, populating
  /// result with the refined map_T_odom.
  bool global_registration(const PointCloud::ConstPtr& scan,
                           const PointCloud::ConstPtr& map,
                           const Eigen::Matrix4d& odom_T_imu,
                           Eigen::Matrix4d& result);

  /// Perform local registration of scan (odom frame) against map (map frame).
  /// Uses map_T_odom_ as initial guess. Returns true on success, populating
  /// result with the refined map_T_odom.
  bool local_registration(const PointCloud::ConstPtr& scan,
                          const PointCloud::ConstPtr& map,
                          const Eigen::Matrix4d& odom_T_imu,
                          Eigen::Matrix4d& result);

  // --- Callbacks ---
  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void timer_callback();
  void initialpose_callback(
      const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);

  // --- Helpers ---

  /// Attempt global registration with the latest scan. Throws on failure.
  void attempt_global_registration();

  /// Look up the odom -> imu transform from tf2.
  bool lookup_odom_T_imu(Eigen::Matrix4d& odom_T_imu,
                          const rclcpp::Time& stamp);

  /// Publish the map -> odom transform via tf2.
  void publish_map_to_odom(const Eigen::Matrix4d& map_T_odom,
                           const rclcpp::Time& stamp);

  /// Compute robust ground height from map cloud at a given (x, y) location.
  double compute_ground_height(double x, double y) const;

  /// Convert (x, y, z, roll, pitch, yaw) to a 4x4 homogeneous transform.
  static Eigen::Matrix4d pose_to_matrix(double x, double y, double z,
                                        double roll, double pitch, double yaw);

  /// Rebuild submap if sensor has moved beyond threshold.
  void maybe_rebuild_submap(const Eigen::Vector3d& sensor_map);

  // --- State machine ---
  State state_{State::WAITING_FOR_TF};

  // --- Point clouds ---
  PointCloud::Ptr map_cloud_;
  std::mutex scan_mutex_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_scan_;

  // --- Submap ---
  PointCloud::Ptr submap_;
  Eigen::Vector2d submap_center_{0.0, 0.0};
  double submap_half_extent_{50.0};
  double submap_rebuild_threshold_{20.0};
  bool submap_valid_{false};

  // --- Transforms ---
  Eigen::Matrix4d map_T_odom_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d initial_guess_{Eigen::Matrix4d::Identity()};

  // --- Matching algorithm ---
  std::unique_ptr<MatchingAlgorithm> matching_algorithm_;

  // --- ROS interfaces ---
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      sub_initialpose_;
  rclcpp::TimerBase::SharedPtr registration_timer_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // --- Parameters ---
  std::string pcd_file_path_;
  std::string lidar_topic_;
  std::string map_frame_;
  std::string odom_frame_;
  std::string body_frame_;
  std::string imu_frame_;
  double timer_period_s_;
  double scan_max_range_{20.0};
  double local_map_radius_{30.0};

  // --- Initial guess ---
  bool use_default_guess_{false};
  bool have_initial_guess_{false};
  double default_roll_{0.0};
  double default_pitch_{0.0};
  double default_z_{0.0};
  double ground_search_radius_x_{5.0};
  double ground_search_radius_y_{5.0};
  double ground_percentile_{0.05};
  bool registration_timing_{false};
  bool correct_roll_pitch_{true};
};

}  // namespace scan_lock
