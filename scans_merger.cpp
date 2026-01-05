/*
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2017, Poznan University of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Poznan University of Technology nor the names
 *       of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Author: Mateusz Przybyla
 */

#include "landmark_detector/scans_merger.h"

using namespace landmark_detector;
using namespace std;

ScansMerger::ScansMerger(ros::NodeHandle& nh, ros::NodeHandle& nh_local) : nh_(nh), nh_local_(nh_local), tf_listener_(tf_buffer_) {
  p_active_ = false;

  front_scan_received_ = false;
  rear_scan_received_ = false;

  front_scan_error_ = false;
  rear_scan_error_ = false;

  params_srv_ = nh_local_.advertiseService("params", &ScansMerger::updateParams, this);

  initialize();
}

ScansMerger::~ScansMerger() {
  nh_local_.deleteParam("active");
  nh_local_.deleteParam("publish_scan");

  nh_local_.deleteParam("ranges_num");

  nh_local_.deleteParam("min_scanner_range");
  nh_local_.deleteParam("max_scanner_range");

  nh_local_.deleteParam("min_x_range");
  nh_local_.deleteParam("max_x_range");
  nh_local_.deleteParam("min_y_range");
  nh_local_.deleteParam("max_y_range");

  nh_local_.deleteParam("fixed_frame_id");
  nh_local_.deleteParam("target_frame_id");
}

bool ScansMerger::updateParams(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res) {
  bool prev_active = p_active_;

  nh_local_.param<bool>("active", p_active_, true);
  nh_local_.param<bool>("publish_scan", p_publish_scan_, true);

  nh_local_.param<int>("ranges_num", p_ranges_num_, 1000);

  nh_local_.param<double>("min_scanner_range", p_min_scanner_range_, 0.05);
  nh_local_.param<double>("max_scanner_range", p_max_scanner_range_, 10.0);

  nh_local_.param<double>("min_x_range", p_min_x_range_, -10.0);
  nh_local_.param<double>("max_x_range", p_max_x_range_,  10.0);
  nh_local_.param<double>("min_y_range", p_min_y_range_, -10.0);
  nh_local_.param<double>("max_y_range", p_max_y_range_,  10.0);
  nh_local_.param<double>("angle_min", angle_min_, -M_PI);
  nh_local_.param<double>("angle_max", angle_max_, M_PI);
  nh_local_.param<double>("angle_increment", angle_increment_, (angle_max_ - angle_min_) / p_ranges_num_);

  nh_local_.param<string>("target_frame_id", p_target_frame_id_, "base_footprint");

  if (p_active_ != prev_active) {
    if (p_active_) {
      front_scan_sub_ = nh_.subscribe("scan_front", 10, &ScansMerger::frontScanCallback, this);
      rear_scan_sub_ = nh_.subscribe("scan_back", 10, &ScansMerger::rearScanCallback, this);
      scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>("scan", 10);
    }
    else {
      front_scan_sub_.shutdown();
      rear_scan_sub_.shutdown();
      scan_pub_.shutdown();
      pcl_pub_.shutdown();
    }
  }

  return true;
}

void ScansMerger::frontScanCallback(const sensor_msgs::LaserScan::ConstPtr& front_scan) {
  std::lock_guard<std::mutex> lock(mutex_);
  scan_front_ = front_scan;
  publishMessages();
  // front_scan_received_ = true;
}

void ScansMerger::rearScanCallback(const sensor_msgs::LaserScan::ConstPtr& rear_scan) {
  std::lock_guard<std::mutex> lock(mutex_);
  scan_back_ = rear_scan;
  publishMessages();
  // rear_scan_received_ = true;
}

void ScansMerger::publishMessages() {
  if (!scan_front_ || !scan_back_)
    return;

  sensor_msgs::LaserScan merged;
  merged.header.stamp = ros::Time::now();
  merged.header.frame_id = p_target_frame_id_;
  merged.angle_min = angle_min_;
  merged.angle_max = angle_max_;
  merged.angle_increment = angle_increment_;
  merged.range_min = std::min(scan_front_->range_min, scan_back_->range_min);
  merged.range_max = std::max(scan_front_->range_max, scan_back_->range_max);

  int bins = std::ceil((angle_max_ - angle_min_) / angle_increment_);
  merged.ranges.assign(bins, std::numeric_limits<float>::infinity());
  merged.intensities.assign(bins, 0.0);

  processScan(scan_front_, merged);
  processScan(scan_back_, merged);
  // ROS_WARN("Processed scans");
  scan_pub_.publish(merged);
  // front_scan_received_ = false;
  // rear_scan_received_ = false;
}

  void ScansMerger::processScan(const sensor_msgs::LaserScan::ConstPtr& scan,
                     sensor_msgs::LaserScan& merged)
    {
        // ROS_WARN("Processing scan from frame: %s", scan->header.frame_id.c_str());
        for (size_t i = 0; i < scan->ranges.size(); ++i)
        {
            float r = scan->ranges[i];
            if (!std::isfinite(r))
                continue;

            double angle = scan->angle_min + i * scan->angle_increment;

            geometry_msgs::PointStamped p_scan, p_base;
            p_scan.header = scan->header;
            p_scan.point.x = r * cos(angle);
            p_scan.point.y = r * sin(angle);
            p_scan.point.z = 0.0;

            try
            {
                tf_buffer_.transform(p_scan, p_base, p_target_frame_id_, ros::Duration(0.05));
            }
            catch (tf2::TransformException &ex)
            {
                ROS_WARN_THROTTLE(1.0, "TF error: %s", ex.what());
                continue;
            }

            double r_base = hypot(p_base.point.x, p_base.point.y);
            double a_base = atan2(p_base.point.y, p_base.point.x);

            if (a_base < angle_min_ || a_base > angle_max_)
                continue;

            int idx = (a_base - angle_min_) / angle_increment_;
            if (idx < 0 || idx >= (int)merged.ranges.size())
                continue;

            if (r_base < merged.ranges[idx])
            {
                merged.ranges[idx] = r_base;
                if (i < scan->intensities.size())
                    merged.intensities[idx] = scan->intensities[i];
            }
        }
    }