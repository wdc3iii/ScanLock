#include "scan_lock.h"

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2/utils.h>

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
  body_frame_ = declare_parameter<std::string>("frames.body_frame", "body");
  imu_frame_ = declare_parameter<std::string>("frames.imu_frame", "imu");
  lidar_topic_ = declare_parameter<std::string>("topics.lidar_topic", "cloud_registered");
  double map_voxel_size = declare_parameter<double>("scan_lock.map_viz_voxel_size", 0.5);

  double x = declare_parameter<double>("initial_guess.x", 0.0);
  double y = declare_parameter<double>("initial_guess.y", 0.0);
  double z = declare_parameter<double>("initial_guess.z", 0.0);
  double roll = declare_parameter<double>("initial_guess.roll", 0.0);
  double pitch = declare_parameter<double>("initial_guess.pitch", 0.0);
  double yaw = declare_parameter<double>("initial_guess.yaw", 0.0);

  default_roll_ = roll;
  default_pitch_ = pitch;
  default_z_ = z;

  use_default_guess_ = declare_parameter<bool>("initial_guess.use_default", false);
  ground_search_radius_x_ = declare_parameter<double>("initial_guess.ground_search_radius_x", 5.0);
  ground_search_radius_y_ = declare_parameter<double>("initial_guess.ground_search_radius_y", 5.0);
  ground_percentile_ = declare_parameter<double>("initial_guess.ground_percentile", 0.05);

  if (use_default_guess_) {
    initial_guess_ = pose_to_matrix(x, y, z, roll, pitch, yaw);
    have_initial_guess_ = true;
  }

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

  // Publish downsampled map for visualization (latched so RViz gets it on subscribe)
  {
    auto qos = rclcpp::QoS(1).transient_local();
    pub_map_ = create_publisher<sensor_msgs::msg::PointCloud2>("map_cloud", qos);

    auto downsampled = std::make_shared<PointCloud>();
    pcl::VoxelGrid<PointType> voxel;
    voxel.setInputCloud(map_cloud_);
    voxel.setLeafSize(map_voxel_size, map_voxel_size, map_voxel_size);
    voxel.filter(*downsampled);

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*downsampled, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
    pub_map_->publish(msg);

    RCLCPP_INFO(get_logger(), "Published downsampled map (%zu -> %zu points, voxel %.2fm)",
                map_cloud_->size(), downsampled->size(), map_voxel_size);
  }

  // Set up tf2
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Publish initial identity map -> odom so the TF tree is defined immediately
  publish_map_to_odom(Eigen::Matrix4d::Identity(), this->now());


  // Subscribe to lidar point cloud
  sub_lidar_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ScanLockNode::lidar_callback, this, std::placeholders::_1));

  // Subscribe to /initialpose (RViz 2D Pose Estimate) for interactive pose setting
  sub_initialpose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/initialpose", rclcpp::QoS(1),
      std::bind(&ScanLockNode::initialpose_callback, this, std::placeholders::_1));

  // Create registration timer
  registration_timer_ = create_wall_timer(
      std::chrono::duration<double>(timer_period_s_),
      std::bind(&ScanLockNode::timer_callback, this));

  if (have_initial_guess_) {
    RCLCPP_INFO(get_logger(), "ScanLock initialized (config initial guess). Waiting for %s -> %s...",
                odom_frame_.c_str(), body_frame_.c_str());
  } else {
    RCLCPP_INFO(get_logger(), "ScanLock initialized. Waiting for /initialpose and %s -> %s...",
                odom_frame_.c_str(), body_frame_.c_str());
  }
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
  //   scan          - current lidar scan (in body frame)
  //   map           - pre-built map point cloud (in map frame)
  //   initial_guess - 4x4 transform: initial estimate of map_T_body
  //
  // Output:
  //   result - refined 4x4 map_T_body transform
  //
  // Return true on success, false on failure.

  // TODO: update roll/pitch of initial guess to reflect actual roll and pitch of the robot. This assumes that the map is gravity aligned. (make a flag for this in the config)
  // note that the map is made such that the LiDAR frame is level, not the body frame. So you may need to add the the lidar frame here, and compute the roll and pitch
  // so that the lidar frame is level with the ground.
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
  //   scan          - current lidar scan (in body frame)
  //   map           - pre-built map point cloud (in map frame)
  //   initial_guess - 4x4 transform: current estimate of map_T_body
  //
  // Output:
  //   result - refined 4x4 map_T_body transform
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
      Eigen::Matrix4d odom_T_body;
      if (lookup_odom_to_body(odom_T_body, msg->header.stamp)) {
        if (have_initial_guess_) {
          RCLCPP_INFO(get_logger(), "TF %s -> %s available. Attempting global registration...",
                      odom_frame_.c_str(), body_frame_.c_str());
          state_ = State::GLOBAL_REGISTRATION;
          attempt_global_registration();
        } else {
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
              "TF available. Waiting for /initialpose...");
        }
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "Waiting for TF %s -> %s...", odom_frame_.c_str(), body_frame_.c_str());
      }
      break;
    }
    case State::GLOBAL_REGISTRATION:
      if (have_initial_guess_) {
        attempt_global_registration();
      }
      break;
    case State::LOCALIZED:
      // Rebroadcast map -> odom at lidar rate so RViz can resolve the frame
      publish_map_to_odom(map_T_odom_, msg->header.stamp);
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
  auto scan_imu = std::make_shared<PointCloud>();
  pcl::fromROSMsg(*scan_msg, *scan_imu);

  // Transform scan from imu frame to body frame
  auto scan = transform_scan_to_body(scan_imu);
  if (!scan) {
    RCLCPP_WARN(get_logger(), "Failed to transform scan to body frame");
    return;
  }

  // Look up current odom -> body transform
  Eigen::Matrix4d odom_T_body;
  if (!lookup_odom_to_body(odom_T_body, scan_msg->header.stamp)) {
    RCLCPP_WARN(get_logger(), "Lost TF %s -> %s during local registration",
                odom_frame_.c_str(), body_frame_.c_str());
    return;
  }

  // Current estimate of body in map: map_T_odom * odom_T_body
  Eigen::Matrix4d current_guess = map_T_odom_ * odom_T_body;

  Eigen::Matrix4d result;
  if (local_registration(scan, map_cloud_, current_guess, result)) {
    map_T_body_ = result;
    map_T_odom_ = map_T_body_ * odom_T_body.inverse();
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
  auto scan_imu = std::make_shared<PointCloud>();
  pcl::fromROSMsg(*scan_msg, *scan_imu);

  // Transform scan from imu frame to body frame
  auto scan = transform_scan_to_body(scan_imu);
  if (!scan) {
    RCLCPP_WARN(get_logger(), "Failed to transform scan to body frame during global registration");
    state_ = State::WAITING_FOR_TF;
    return;
  }

  // Look up odom -> body
  Eigen::Matrix4d odom_T_body;
  if (!lookup_odom_to_body(odom_T_body, scan_msg->header.stamp)) {
    RCLCPP_WARN(get_logger(), "TF lookup failed during global registration attempt");
    state_ = State::WAITING_FOR_TF;
    return;
  }

  Eigen::Matrix4d result;
  if (global_registration(scan, map_cloud_, initial_guess_, result)) {
    map_T_body_ = result;
    map_T_odom_ = map_T_body_ * odom_T_body.inverse();
    publish_map_to_odom(map_T_odom_, scan_msg->header.stamp);
    state_ = State::LOCALIZED;
    RCLCPP_INFO(get_logger(), "Global registration succeeded. Localized in map.");
  } else {
    RCLCPP_FATAL(get_logger(), "Global registration failed!");
    throw std::runtime_error("Global registration failed");
  }
}

bool ScanLockNode::lookup_odom_to_body(Eigen::Matrix4d& odom_T_body,
                                       const rclcpp::Time& stamp) {
  try {
    auto transform = tf_buffer_->lookupTransform(
        odom_frame_, body_frame_, stamp,
        rclcpp::Duration::from_seconds(0.1));

    Eigen::Isometry3d eigen_tf = tf2::transformToEigen(transform);
    odom_T_body = eigen_tf.matrix();
    return true;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_DEBUG(get_logger(), "TF lookup failed: %s", ex.what());
    return false;
  }
}

bool ScanLockNode::lookup_body_T_imu() {
  try {
    auto transform = tf_buffer_->lookupTransform(
        body_frame_, imu_frame_, rclcpp::Time(0),
        rclcpp::Duration::from_seconds(1.0));

    Eigen::Isometry3d eigen_tf = tf2::transformToEigen(transform);
    body_T_imu_ = eigen_tf.matrix();
    have_body_T_imu_ = true;
    RCLCPP_INFO(get_logger(), "Cached %s -> %s transform",
                body_frame_.c_str(), imu_frame_.c_str());
    return true;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_DEBUG(get_logger(), "body_T_imu lookup failed: %s", ex.what());
    return false;
  }
}

PointCloud::Ptr ScanLockNode::transform_scan_to_body(
    const PointCloud::ConstPtr& scan_imu) {
  if (!have_body_T_imu_) {
    if (!lookup_body_T_imu()) {
      return nullptr;
    }
  }
  auto scan_body = std::make_shared<PointCloud>();
  Eigen::Matrix4f body_T_imu_f = body_T_imu_.cast<float>();
  pcl::transformPointCloudWithNormals(*scan_imu, *scan_body, body_T_imu_f);
  return scan_body;
}

void ScanLockNode::initialpose_callback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg) {
  double x = msg->pose.pose.position.x;
  double y = msg->pose.pose.position.y;

  // Extract yaw from quaternion
  tf2::Quaternion q(
      msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  double roll_unused, pitch_unused, yaw;
  tf2::Matrix3x3(q).getRPY(roll_unused, pitch_unused, yaw);

  // Compute ground height from map, add default body height offset
  double ground_z = compute_ground_height(x, y);
  double z = ground_z + default_z_;

  // x/y/yaw from RViz, roll/pitch from config defaults,
  // z from ground estimation + config offset
  initial_guess_ = pose_to_matrix(x, y, z, default_roll_, default_pitch_, yaw);
  have_initial_guess_ = true;

  RCLCPP_INFO(get_logger(),
      "Initial pose from RViz: x=%.2f y=%.2f z=%.2f (ground=%.2f + offset=%.2f) yaw=%.2f rad",
      x, y, z, ground_z, default_z_, yaw);

  // Trigger (re-)localization
  if (state_ == State::LOCALIZED || state_ == State::GLOBAL_REGISTRATION) {
    state_ = State::GLOBAL_REGISTRATION;
    RCLCPP_INFO(get_logger(), "Re-entering global registration with new initial pose.");
  }
}

double ScanLockNode::compute_ground_height(double x, double y) const {
  std::vector<float> z_values;
  z_values.reserve(1000);

  for (const auto& pt : map_cloud_->points) {
    if (std::abs(pt.x - static_cast<float>(x)) < ground_search_radius_x_ &&
        std::abs(pt.y - static_cast<float>(y)) < ground_search_radius_y_) {
      z_values.push_back(pt.z);
    }
  }

  if (z_values.empty()) {
    RCLCPP_WARN(get_logger(),
        "No map points found near (%.2f, %.2f) within (%.1f x %.1f) region. Using z=0.",
        x, y, ground_search_radius_x_, ground_search_radius_y_);
    return 0.0;
  }

  size_t idx = static_cast<size_t>(ground_percentile_ * (z_values.size() - 1));
  std::nth_element(z_values.begin(), z_values.begin() + idx, z_values.end());
  double ground_z = z_values[idx];

  RCLCPP_INFO(get_logger(),
      "Ground height at (%.2f, %.2f): %.3f (%zu points in search region)",
      x, y, ground_z, z_values.size());
  return ground_z;
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
