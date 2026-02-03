#pragma once

#include <mutex>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "frame.h"

/**
 * VisualOdometry Class - Performs monocular visual odometry.
 * Processes incoming frames to detect and match features, estimates
 * relative camera motion between frames, and maintains a 3D trajectory.
 */
class VisualOdometry {
public:
    VisualOdometry(int image_width, int image_height);

    cv::Mat process_frame(Frame& frame);
    std::vector<cv::Mat> get_trajectory();

private:
    void detect_features(Frame& frame);
    std::vector<cv::DMatch> match_features(const Frame& frame1, const Frame& frame2);

    bool estimate_relative_pose(
        const Frame& frame1,
        const Frame& frame2,
        const std::vector<cv::DMatch>& matches,
        cv::Mat& R,
        cv::Mat& t
    );

    cv::Ptr<cv::ORB> orb_detector_;
    cv::Ptr<cv::BFMatcher> matcher_;
    cv::Mat camera_matrix_;

    std::vector<cv::Mat> trajectory_positions_;
    std::mutex trajectory_mutex_;

    Frame previous_frame_;
    bool initialized_ = false;
};
