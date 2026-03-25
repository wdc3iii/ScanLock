#include "scan_lock.h"

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

namespace scan_lock {

ScanLockNode::ScanLockNode(const rclcpp::NodeOptions& options)
    : Node("scan_lock", options),
      map_cloud_(std::make_shared<PointCloud>()) {
  // Declare parameters
  std::string pcd_file_name =
      declare_parameter<std::string>("scan_lock.pcd_file_name", "");
  timer_period_s_ = declare_parameter<double>("scan_lock.register_period_s", 1.0);
  map_frame_ = declare_parameter<std::string>("frames.map_frame", "map");
  odom_frame_ = declare_parameter<std::string>("frames.odom_frame", "odom");
  lidar_frame_ = declare_parameter<std::string>("frames.lidar_frame", "lidar");
  lidar_topic_ = declare_parameter<std::string>("topics.lidar_topic", "cloud_registered");

  double x = declare_parameter<double>("initial_guess.x", 0.0);
  double y = declare_parameter<double>("initial_guess.y", 0.0);
  double z = declare_parameter<double>("initial_guess.z", 0.0);
  double roll = declare_parameter<double>("initial_guess.roll", 0.0);
  double pitch = declare_parameter<double>("initial_guess.pitch", 0.0);
  double yaw = declare_parameter<double>("initial_guess.yaw", 0.0);
  initial_guess_ = pose_to_matrix(x, y, z, roll, pitch, yaw);

  // Load the point cloud map from pcd/ directory
  if (pcd_file_name.empty()) {
    RCLCPP_FATAL(get_logger(), "No PCD file name specified (scan_lock.pcd_file_name)");
    throw std::runtime_error("No PCD file name specified");
  }

  pcd_file_path_ = std::string(ROOT_DIR) + "pcd/" + pcd_file_name;
  if (pcl::io::loadPCDFile<PointType>(pcd_file_path_, *map_cloud_) == -1) {
    RCLCPP_FATAL(get_logger(), "Failed to load PCD file: %s", pcd_file_path_.c_str());
    throw std::runtime_error("Failed to load PCD file: " + pcd_file_path_);
  }
  RCLCPP_INFO(get_logger(), "Loaded map with %zu points from %s",
              map_cloud_->size(), pcd_file_path_.c_str());

  // Set up tf2
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Subscribe to lidar point cloud
  sub_lidar_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ScanLockNode::lidar_callback, this, std::placeholders::_1));

  // Create registration timer
  registration_timer_ = create_wall_timer(
      std::chrono::duration<double>(timer_period_s_),
      std::bind(&ScanLockNode::timer_callback, this));

  RCLCPP_INFO(get_logger(), "ScanLock initialized. Waiting for %s -> %s transform...",
              odom_frame_.c_str(), lidar_frame_.c_str());
}

// ---------------------------------------------------------------------------
// Registration stubs
// ---------------------------------------------------------------------------

bool ScanLockNode::global_registration(
    const PointCloud::ConstPtr& scan,
    const PointCloud::ConstPtr& map,
    const Eigen::Matrix4d& initial_guess,
    Eigen::Matrix4d& result) {
  // TODO: Implement global point cloud registration (e.g., FPFH + RANSAC, NDT
  // with wide search, or other coarse alignment method).
  //
  // Inputs:
  //   scan          - current lidar scan (in lidar frame)
  //   map           - pre-built map point cloud (in map frame)
  //   initial_guess - 4x4 transform: initial estimate of map_T_lidar
  //
  // Output:
  //   result - refined 4x4 map_T_lidar transform
  //
  // Return true on success, false on failure.

  local_registration(scan, map, initial_guess, result);
  return true;
}

bool ScanLockNode::local_registration(
    const PointCloud::ConstPtr& /*scan*/,
    const PointCloud::ConstPtr& /*map*/,
    const Eigen::Matrix4d& initial_guess,
    Eigen::Matrix4d& result) {
  // TODO: Implement local point cloud registration (e.g., ICP, NDT with tight
  // convergence criteria).
  //
  // Inputs:
  //   scan          - current lidar scan (in lidar frame)
  //   map           - pre-built map point cloud (in map frame)
  //   initial_guess - 4x4 transform: current estimate of map_T_lidar
  //
  // Output:
  //   result - refined 4x4 map_T_lidar transform
  //
  // Return true on success, false on failure.

  result = initial_guess;
  return true;
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void ScanLockNode::lidar_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    latest_scan_ = msg;
  }

  switch (state_) {
    case State::WAITING_FOR_TF: {
      Eigen::Matrix4d odom_T_lidar;
      if (lookup_odom_to_lidar(odom_T_lidar, msg->header.stamp)) {
        RCLCPP_INFO(get_logger(), "TF %s -> %s available. Attempting global registration...",
                    odom_frame_.c_str(), lidar_frame_.c_str());
        state_ = State::GLOBAL_REGISTRATION;
        attempt_global_registration();
      } else {
        RCLCPP_WARN(get_logger(), "Waiting for TF %s -> %s to attempt global registration...",
                odom_frame_.c_str(), lidar_frame_.c_str());
      }
      break;
    }
    case State::GLOBAL_REGISTRATION:
      attempt_global_registration();
      break;
    case State::LOCALIZED:
      // Timer handles local registration
      break;
  }
}

void ScanLockNode::timer_callback() {
  if (state_ != State::LOCALIZED) {
    return;
  }

  sensor_msgs::msg::PointCloud2::ConstSharedPtr scan_msg;
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    scan_msg = latest_scan_;
  }

  if (!scan_msg) {
    return;
  }

  // Convert ROS message to PCL
  auto scan = std::make_shared<PointCloud>();
  pcl::fromROSMsg(*scan_msg, *scan);

  // Look up current odom -> lidar transform
  Eigen::Matrix4d odom_T_lidar;
  if (!lookup_odom_to_lidar(odom_T_lidar, scan_msg->header.stamp)) {
    RCLCPP_WARN(get_logger(), "Lost TF %s -> %s during local registration",
                odom_frame_.c_str(), lidar_frame_.c_str());
    return;
  }

  // Current estimate of lidar in map: map_T_odom * odom_T_lidar
  Eigen::Matrix4d current_guess = map_T_odom_ * odom_T_lidar;

  Eigen::Matrix4d result;
  if (local_registration(scan, map_cloud_, current_guess, result)) {
    map_T_lidar_ = result;
    map_T_odom_ = map_T_lidar_ * odom_T_lidar.inverse();
    publish_map_to_odom(map_T_odom_, scan_msg->header.stamp);
    RCLCPP_INFO(get_logger(), "Local registration succeeded. Updated pose in map.");
  } else {
    RCLCPP_WARN(get_logger(),
                "Local registration failed. Falling back to global registration.");
    state_ = State::GLOBAL_REGISTRATION;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void ScanLockNode::attempt_global_registration() {
  sensor_msgs::msg::PointCloud2::ConstSharedPtr scan_msg;
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    scan_msg = latest_scan_;
  }

  if (!scan_msg) {
    return;
  }

  // Convert ROS message to PCL
  auto scan = std::make_shared<PointCloud>();
  pcl::fromROSMsg(*scan_msg, *scan);

  // Look up odom -> lidar
  Eigen::Matrix4d odom_T_lidar;
  if (!lookup_odom_to_lidar(odom_T_lidar, scan_msg->header.stamp)) {
    RCLCPP_WARN(get_logger(), "TF lookup failed during global registration attempt");
    state_ = State::WAITING_FOR_TF;
    return;
  }

  Eigen::Matrix4d result;
  if (global_registration(scan, map_cloud_, initial_guess_, result)) {
    map_T_lidar_ = result;
    map_T_odom_ = map_T_lidar_ * odom_T_lidar.inverse();
    publish_map_to_odom(map_T_odom_, scan_msg->header.stamp);
    state_ = State::LOCALIZED;
    RCLCPP_INFO(get_logger(), "Global registration succeeded. Localized in map.");
  } else {
    RCLCPP_FATAL(get_logger(), "Global registration failed!");
    throw std::runtime_error("Global registration failed");
  }
}

bool ScanLockNode::lookup_odom_to_lidar(Eigen::Matrix4d& odom_T_lidar,
                                        const rclcpp::Time& stamp) {
  try {
    auto transform = tf_buffer_->lookupTransform(
        odom_frame_, lidar_frame_, stamp,
        rclcpp::Duration::from_seconds(0.1));

    Eigen::Isometry3d eigen_tf = tf2::transformToEigen(transform);
    odom_T_lidar = eigen_tf.matrix();
    return true;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_DEBUG(get_logger(), "TF lookup failed: %s", ex.what());
    return false;
  }
}

void ScanLockNode::publish_map_to_odom(const Eigen::Matrix4d& map_T_odom,
                                       const rclcpp::Time& stamp) {
  Eigen::Isometry3d iso(map_T_odom);
  geometry_msgs::msg::TransformStamped tf_msg = tf2::eigenToTransform(iso);
  tf_msg.header.stamp = stamp;
  tf_msg.header.frame_id = map_frame_;
  tf_msg.child_frame_id = odom_frame_;
  tf_broadcaster_->sendTransform(tf_msg);
}

Eigen::Matrix4d ScanLockNode::pose_to_matrix(double x, double y, double z,
                                             double roll, double pitch,
                                             double yaw) {
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.translation() = Eigen::Vector3d(x, y, z);
  transform.linear() =
      (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
  return transform.matrix();
}

}  // namespace scan_lock

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(scan_lock::ScanLockNode)
