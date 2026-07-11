#include "pose_prompt_node.h"

#include <limits>

#include <Eigen/Core>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

PosePromptNode::PosePromptNode(const rclcpp::NodeOptions& options)
    : Node("pose_prompt", options),
      map_cloud_(std::make_shared<PosePromptPointCloud>()) {
  std::string pcd_file_name =
      declare_parameter<std::string>("pose_prompt.pcd_file_name", "");
  double coarse_voxel_size =
      declare_parameter<double>("pose_prompt.coarse_voxel_size", 2.0);
  fine_voxel_size_ = declare_parameter<double>("pose_prompt.fine_voxel_size", 0.05);
  fine_region_radius_ = declare_parameter<double>("pose_prompt.fine_region_radius", 20.0);
  fine_point_cap_ = declare_parameter<int>("pose_prompt.fine_point_cap", 3000000);
  map_frame_ = declare_parameter<std::string>("frames.map_frame", "map");
  std::string region_point_topic =
      declare_parameter<std::string>("topics.region_point_topic", "/clicked_point");

  if (fine_voxel_size_ <= 0.0) {
    RCLCPP_WARN(get_logger(),
        "pose_prompt.fine_voxel_size must be > 0 (got %.4f); clamping to 0.05m. "
        "A 0/full-resolution fine crop can produce multi-million-point latched "
        "messages that stall RViz and the network.",
        fine_voxel_size_);
    fine_voxel_size_ = 0.05;
  }

  if (pcd_file_name.empty()) {
    RCLCPP_FATAL(get_logger(), "No PCD file name specified (pose_prompt.pcd_file_name)");
    throw std::runtime_error("No PCD file name specified");
  }

  std::string pcd_file_path = std::string(ROOT_DIR) + "pcd/" + pcd_file_name;
  if (pcl::io::loadPCDFile<PosePromptPointType>(pcd_file_path, *map_cloud_) == -1) {
    RCLCPP_FATAL(get_logger(), "Failed to load PCD file: %s", pcd_file_path.c_str());
    throw std::runtime_error("Failed to load PCD file: " + pcd_file_path);
  }
  RCLCPP_INFO(get_logger(), "Loaded map with %zu points from %s",
              map_cloud_->size(), pcd_file_path.c_str());

  auto qos = rclcpp::QoS(1).transient_local();
  pub_coarse_ = create_publisher<sensor_msgs::msg::PointCloud2>("map_cloud_coarse", qos);
  pub_fine_ = create_publisher<sensor_msgs::msg::PointCloud2>("map_cloud_fine", qos);

  // Publish the coarse full-map overview once at startup.
  {
    auto coarse = std::make_shared<PosePromptPointCloud>();
    pcl::VoxelGrid<PosePromptPointType> voxel;
    voxel.setInputCloud(map_cloud_);
    voxel.setLeafSize(coarse_voxel_size, coarse_voxel_size, coarse_voxel_size);
    voxel.filter(*coarse);

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*coarse, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
    pub_coarse_->publish(msg);

    RCLCPP_INFO(get_logger(), "Published coarse map overview (%zu -> %zu points, voxel %.2fm)",
                map_cloud_->size(), coarse->size(), coarse_voxel_size);
  }

  sub_region_point_ = create_subscription<geometry_msgs::msg::PointStamped>(
      region_point_topic, rclcpp::QoS(1),
      std::bind(&PosePromptNode::region_point_callback, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(),
      "PosePrompt ready. Use RViz 'Publish Point' on %s to crop a %.0fm fine region, "
      "then '2D Pose Estimate' to publish /initialpose.",
      region_point_topic.c_str(), fine_region_radius_);
}

void PosePromptNode::region_point_callback(
    const geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {
  Eigen::Vector4f min_pt(
      static_cast<float>(msg->point.x - fine_region_radius_),
      static_cast<float>(msg->point.y - fine_region_radius_),
      -std::numeric_limits<float>::max(), 1.0f);
  Eigen::Vector4f max_pt(
      static_cast<float>(msg->point.x + fine_region_radius_),
      static_cast<float>(msg->point.y + fine_region_radius_),
      std::numeric_limits<float>::max(), 1.0f);

  pcl::CropBox<PosePromptPointType> crop;
  crop.setInputCloud(map_cloud_);
  crop.setMin(min_pt);
  crop.setMax(max_pt);
  auto cropped = std::make_shared<PosePromptPointCloud>();
  crop.filter(*cropped);

  if (cropped->empty()) {
    RCLCPP_WARN(get_logger(), "No map points within %.0fm of clicked point (%.2f, %.2f)",
                fine_region_radius_, msg->point.x, msg->point.y);
    return;
  }

  // Downsample; if still too dense, back off to a coarser leaf size rather
  // than shipping an unbounded latched message.
  auto fine = std::make_shared<PosePromptPointCloud>();
  double leaf = fine_voxel_size_;
  constexpr int kMaxAttempts = 5;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    pcl::VoxelGrid<PosePromptPointType> voxel;
    voxel.setInputCloud(cropped);
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.filter(*fine);

    if (static_cast<int>(fine->size()) <= fine_point_cap_) {
      break;
    }
    RCLCPP_WARN(get_logger(),
        "Fine crop has %zu points (> cap %d) at voxel %.3fm; doubling voxel size.",
        fine->size(), fine_point_cap_, leaf);
    leaf *= 2.0;
  }

  sensor_msgs::msg::PointCloud2 out_msg;
  pcl::toROSMsg(*fine, out_msg);
  out_msg.header.frame_id = map_frame_;
  out_msg.header.stamp = this->now();
  pub_fine_->publish(out_msg);

  RCLCPP_INFO(get_logger(),
      "Fine crop at (%.2f, %.2f): %zu -> %zu points (region %.0fm, voxel %.3fm)",
      msg->point.x, msg->point.y, cropped->size(), fine->size(), fine_region_radius_, leaf);
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PosePromptNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
