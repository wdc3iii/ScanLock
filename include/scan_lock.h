#pragma once

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

namespace scan_lock {

using PointType = pcl::PointXYZINormal;
using PointCloud = pcl::PointCloud<PointType>;

enum class State {
  WAITING_FOR_TF,
  GLOBAL_REGISTRATION,
  LOCALIZED
};

class ScanLockNode : public rclcpp::Node {
public:
  explicit ScanLockNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  // --- Registration methods (stubs for now) ---

  /// Perform global registration of scan against map using an initial guess.
  /// Returns true on success, populating result with the map_T_body transform.
  bool global_registration(const PointCloud::ConstPtr& scan,
                           const PointCloud::ConstPtr& map,
                           const Eigen::Matrix4d& initial_guess,
                           Eigen::Matrix4d& result);

  /// Perform local registration using current pose estimate as initial guess.
  /// Returns true on success, populating result with the refined map_T_body.
  bool local_registration(const PointCloud::ConstPtr& scan,
                          const PointCloud::ConstPtr& map,
                          const Eigen::Matrix4d& initial_guess,
                          Eigen::Matrix4d& result);

  // --- Callbacks ---
  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void timer_callback();
  void initialpose_callback(
      const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);

  // --- Helpers ---

  /// Attempt global registration with the latest scan. Throws on failure.
  void attempt_global_registration();

  /// Look up the odom -> body transform from tf2.
  bool lookup_odom_to_body(Eigen::Matrix4d& odom_T_body,
                           const rclcpp::Time& stamp);

  /// Look up the body -> imu transform from tf2 (cached after first success).
  bool lookup_body_T_imu();

  /// Transform a scan from imu frame to body frame. Returns nullptr on failure.
  PointCloud::Ptr transform_scan_to_body(const PointCloud::ConstPtr& scan_imu);

  /// Publish the map -> odom transform via tf2.
  void publish_map_to_odom(const Eigen::Matrix4d& map_T_odom,
                           const rclcpp::Time& stamp);

  /// Compute robust ground height from map cloud at a given (x, y) location.
  double compute_ground_height(double x, double y) const;

  /// Convert (x, y, z, roll, pitch, yaw) to a 4x4 homogeneous transform.
  static Eigen::Matrix4d pose_to_matrix(double x, double y, double z,
                                        double roll, double pitch, double yaw);

  // --- State machine ---
  State state_{State::WAITING_FOR_TF};

  // --- Point clouds ---
  PointCloud::Ptr map_cloud_;
  std::mutex scan_mutex_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_scan_;

  // --- Transforms ---
  Eigen::Matrix4d map_T_body_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d map_T_odom_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d initial_guess_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d body_T_imu_{Eigen::Matrix4d::Identity()};
  bool have_body_T_imu_{false};

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

  // --- Initial guess ---
  bool use_default_guess_{false};
  bool have_initial_guess_{false};
  double default_roll_{0.0};
  double default_pitch_{0.0};
  double default_z_{0.0};
  double ground_search_radius_x_{5.0};
  double ground_search_radius_y_{5.0};
  double ground_percentile_{0.05};
};

}  // namespace scan_lock
