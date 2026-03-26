#include "scan_lock.h"
#include "gicp.h"

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2/utils.h>

namespace scan_lock {

ScanLockNode::ScanLockNode(const rclcpp::NodeOptions& options)
    : Node("scan_lock", options),
      map_cloud_(std::make_shared<PointCloud>()),
      submap_(std::make_shared<PointCloud>()),
      matching_algorithm_(std::make_unique<GICP>()) {
  // Declare parameters
  std::string pcd_file_name =
      declare_parameter<std::string>("scan_lock.pcd_file_name", "");
  timer_period_s_ = declare_parameter<double>("scan_lock.register_period_s", 1.0);
  scan_max_range_ = declare_parameter<double>("scan_lock.scan_max_range", 20.0);
  local_map_radius_ = declare_parameter<double>("scan_lock.local_map_radius", 30.0);
  submap_half_extent_ = declare_parameter<double>("scan_lock.submap_half_extent", 50.0);
  submap_rebuild_threshold_ = declare_parameter<double>("scan_lock.submap_rebuild_threshold", 20.0);

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
  registration_timing_ = declare_parameter<bool>("scan_lock.registration_timing", false);
  correct_roll_pitch_ = declare_parameter<bool>("scan_lock.correct_roll_pitch", true);

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

  // Set up tf2 (static broadcaster -- transform persists until re-published)
  tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Publish initial identity map -> odom so the TF tree is defined immediately
  publish_map_to_odom(Eigen::Matrix4d::Identity());

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
                odom_frame_.c_str(), imu_frame_.c_str());
  } else {
    RCLCPP_INFO(get_logger(), "ScanLock initialized. Waiting for /initialpose and %s -> %s...",
                odom_frame_.c_str(), imu_frame_.c_str());
  }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

bool ScanLockNode::global_registration(
    const PointCloud::ConstPtr& scan,
    const PointCloud::ConstPtr& map,
    const Eigen::Matrix4d& odom_T_imu,
    Eigen::Matrix4d& result) {
  // Global registration delegates to local_registration for now.
  // A future implementation could use FPFH + RANSAC for coarse alignment.
  return local_registration(scan, map, odom_T_imu, result);
}

bool ScanLockNode::local_registration(
    const PointCloud::ConstPtr& scan,
    const PointCloud::ConstPtr& map,
    const Eigen::Matrix4d& odom_T_imu,
    Eigen::Matrix4d& result) {

  auto start = std::chrono::steady_clock::now();

  // 1. Current estimate (initial guess for GICP)
  Eigen::Matrix4d map_T_odom_est = map_T_odom_;

  // 2. Sensor position in odom frame and map frame
  Eigen::Vector3d sensor_odom = odom_T_imu.block<3,1>(0,3);
  Eigen::Vector4d sensor_odom_h;
  sensor_odom_h << sensor_odom, 1.0;
  Eigen::Vector3d sensor_map = (map_T_odom_est * sensor_odom_h).head<3>();

  // 3. Rebuild submap if needed
  maybe_rebuild_submap(sensor_map);

  // 4. Distance-filter scan in odom frame (keep points near sensor)
  auto scan_filtered = std::make_shared<PointCloud>();
  double range_sq = scan_max_range_ * scan_max_range_;
  for (const auto& pt : scan->points) {
    double dx = pt.x - sensor_odom(0);
    double dy = pt.y - sensor_odom(1);
    double dz = pt.z - sensor_odom(2);
    if (dx*dx + dy*dy + dz*dz < range_sq) {
      scan_filtered->points.push_back(pt);
    }
  }
  scan_filtered->width = scan_filtered->points.size();
  scan_filtered->height = 1;
  scan_filtered->is_dense = scan->is_dense;

  if (scan_filtered->empty()) {
    RCLCPP_WARN(get_logger(), "No scan points within %.1fm of sensor", scan_max_range_);
    return false;
  }

  // 5. Crop map (submap) in map frame (keep points near sensor)
  auto map_cropped = std::make_shared<PointCloud>();
  double map_range_sq = local_map_radius_ * local_map_radius_;
  const auto& map_source = submap_valid_ ? submap_ : map;
  for (const auto& pt : map_source->points) {
    double dx = pt.x - sensor_map(0);
    double dy = pt.y - sensor_map(1);
    double dz = pt.z - sensor_map(2);
    if (dx*dx + dy*dy + dz*dz < map_range_sq) {
      map_cropped->points.push_back(pt);
    }
  }
  map_cropped->width = map_cropped->points.size();
  map_cropped->height = 1;
  map_cropped->is_dense = map_source->is_dense;

  if (map_cropped->empty()) {
    RCLCPP_WARN(get_logger(), "No map points within %.1fm of sensor position", local_map_radius_);
    return false;
  }

  // 6. Run matching algorithm
  Eigen::Matrix4f T;
  if (!matching_algorithm_->align(scan_filtered, map_cropped,
                                   map_T_odom_est.cast<float>(), T)) {
    RCLCPP_WARN(get_logger(), "Matching algorithm failed to converge");
    return false;
  }

  // 7. Result is directly the refined map_T_odom
  // TODO: for debugging, just return the estimate (no scan matching).
  // result = map_T_odom_est.cast<double>();
  result = T.cast<double>();


  if (registration_timing_) {
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    RCLCPP_INFO(get_logger(), "Registration: %.3fs (scan=%zu, map=%zu)",
                elapsed, scan_filtered->size(), map_cropped->size());
  }

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
      // Check if odom -> imu TF is available
      Eigen::Matrix4d odom_T_imu;
      if (lookup_odom_T_imu(odom_T_imu, msg->header.stamp)) {
        if (have_initial_guess_) {
          state_ = State::GLOBAL_REGISTRATION;
          RCLCPP_INFO(get_logger(), "TF available. Moving to GLOBAL_REGISTRATION.");
        } else {
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
              "TF available but waiting for /initialpose...");
        }
      }
      break;
    }
    case State::GLOBAL_REGISTRATION:
      if (have_initial_guess_) {
        attempt_global_registration();
      }
      break;
    case State::LOCALIZED:
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

  // Convert ROS message to PCL (scan is in odom frame)
  auto scan_odom = std::make_shared<PointCloud>();
  pcl::fromROSMsg(*scan_msg, *scan_odom);

  // Look up current odom -> imu transform (for sensor position)
  Eigen::Matrix4d odom_T_imu;
  if (!lookup_odom_T_imu(odom_T_imu, scan_msg->header.stamp)) {
    RCLCPP_WARN(get_logger(), "Lost TF %s -> %s during local registration",
                odom_frame_.c_str(), imu_frame_.c_str());
    return;
  }

  Eigen::Matrix4d result;
  if (local_registration(scan_odom, map_cloud_, odom_T_imu, result)) {
    map_T_odom_ = result;
    publish_map_to_odom(map_T_odom_);
    RCLCPP_DEBUG(get_logger(), "Local registration succeeded. Updated map_T_odom.");
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

  // Convert ROS message to PCL (scan is in odom frame)
  auto scan_odom = std::make_shared<PointCloud>();
  pcl::fromROSMsg(*scan_msg, *scan_odom);

  // Look up odom -> body (for initial guess computation)
  Eigen::Matrix4d odom_T_body;
  if (!lookup_odom_T_body(odom_T_body, scan_msg->header.stamp)) {
    RCLCPP_WARN(get_logger(), "TF lookup failed for odom -> body");
    state_ = State::WAITING_FOR_TF;
    return;
  }

  // Look up odom -> imu (needed for sensor position in registration)
  Eigen::Matrix4d odom_T_imu;
  if (!lookup_odom_T_imu(odom_T_imu, scan_msg->header.stamp)) {
    RCLCPP_WARN(get_logger(), "TF lookup failed for odom -> imu");
    state_ = State::WAITING_FOR_TF;
    return;
  }

  // initial_guess_ is map_T_body from RViz (or config).
  // Correct roll/pitch if enabled.
  Eigen::Matrix4d map_T_body_guess = initial_guess_;

  if (correct_roll_pitch_) {
    // Extract roll/pitch from odom_T_body (odom is gravity-aligned)
    Eigen::Matrix3d R_odom_body = odom_T_body.block<3,3>(0,0);
    double odom_roll = std::atan2(R_odom_body(2,1), R_odom_body(2,2));
    double odom_pitch = std::atan2(-R_odom_body(2,0),
        std::sqrt(R_odom_body(2,1)*R_odom_body(2,1) + R_odom_body(2,2)*R_odom_body(2,2)));

    // Extract yaw from initial_guess_ and keep translation
    Eigen::Matrix3d R_guess = map_T_body_guess.block<3,3>(0,0);
    double guess_yaw = std::atan2(R_guess(1,0), R_guess(0,0));

    // Reconstruct rotation with corrected roll/pitch
    Eigen::Matrix3d R_corrected =
        (Eigen::AngleAxisd(guess_yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(odom_pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(odom_roll, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();

    map_T_body_guess.block<3,3>(0,0) = R_corrected;

    RCLCPP_INFO(get_logger(),
        "Corrected initial guess roll/pitch from body odometry: roll=%.2f deg, pitch=%.2f deg",
        odom_roll * 180.0 / M_PI, odom_pitch * 180.0 / M_PI);
  }

  // Convert map_T_body to map_T_odom:
  //   map_T_body = map_T_odom * odom_T_body
  //   => map_T_odom = map_T_body * odom_T_body^-1
  map_T_odom_ = map_T_body_guess * odom_T_body.inverse();

  Eigen::Matrix4d result;
  if (global_registration(scan_odom, map_cloud_, odom_T_imu, result)) {
    map_T_odom_ = result;
    publish_map_to_odom(map_T_odom_);
    state_ = State::LOCALIZED;
    RCLCPP_INFO(get_logger(), "Global registration succeeded. Localized in map.");
  } else {
    RCLCPP_FATAL(get_logger(), "Global registration failed!");
    throw std::runtime_error("Global registration failed");
  }
}

bool ScanLockNode::lookup_odom_T_imu(Eigen::Matrix4d& odom_T_imu,
                                      const rclcpp::Time& stamp) {
  try {
    auto transform = tf_buffer_->lookupTransform(
        odom_frame_, imu_frame_, stamp,
        rclcpp::Duration::from_seconds(0.1));

    Eigen::Isometry3d eigen_tf = tf2::transformToEigen(transform);
    odom_T_imu = eigen_tf.matrix();
    return true;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_DEBUG(get_logger(), "TF lookup failed: %s", ex.what());
    return false;
  }
}

bool ScanLockNode::lookup_odom_T_body(Eigen::Matrix4d& odom_T_body,
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

  // initial_guess_ is map_T_body: x/y/yaw from RViz, roll/pitch from config
  // defaults, z from ground estimation + config offset
  initial_guess_ = pose_to_matrix(x, y, z, default_roll_, default_pitch_, yaw);
  have_initial_guess_ = true;

  RCLCPP_INFO(get_logger(),
      "Initial pose from RViz: x=%.2f y=%.2f z=%.2f (ground=%.2f + offset=%.2f) yaw=%.2f rad",
      x, y, z, ground_z, default_z_, yaw);

  // Trigger (re-)localization
  if (state_ == State::LOCALIZED || state_ == State::GLOBAL_REGISTRATION) {
    state_ = State::GLOBAL_REGISTRATION;
    submap_valid_ = false;  // Force submap rebuild on re-localization
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

void ScanLockNode::publish_map_to_odom(const Eigen::Matrix4d& map_T_odom) {
  Eigen::Isometry3d iso(map_T_odom);
  geometry_msgs::msg::TransformStamped tf_msg = tf2::eigenToTransform(iso);
  tf_msg.header.stamp = this->now();
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

void ScanLockNode::maybe_rebuild_submap(const Eigen::Vector3d& sensor_map) {
  Eigen::Vector2d sensor_xy(sensor_map(0), sensor_map(1));

  if (submap_valid_) {
    Eigen::Vector2d delta = sensor_xy - submap_center_;
    if (std::abs(delta(0)) < submap_rebuild_threshold_ &&
        std::abs(delta(1)) < submap_rebuild_threshold_) {
      return;  // Submap still valid
    }
  }

  // Rebuild submap
  submap_->clear();
  float cx = static_cast<float>(sensor_xy(0));
  float cy = static_cast<float>(sensor_xy(1));
  float half = static_cast<float>(submap_half_extent_);

  for (const auto& pt : map_cloud_->points) {
    if (std::abs(pt.x - cx) < half && std::abs(pt.y - cy) < half) {
      submap_->points.push_back(pt);
    }
  }
  submap_->width = submap_->points.size();
  submap_->height = 1;
  submap_->is_dense = map_cloud_->is_dense;
  submap_center_ = sensor_xy;
  submap_valid_ = true;

  RCLCPP_INFO(get_logger(), "Rebuilt submap at (%.1f, %.1f): %zu points (half_extent=%.1fm)",
              cx, cy, submap_->size(), submap_half_extent_);
}

}  // namespace scan_lock

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(scan_lock::ScanLockNode)
