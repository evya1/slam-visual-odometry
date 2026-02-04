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

    void print_debugging_statistics(double min_dist, double max_dist, std::size_t num_matches, double mean_dist,
                                    double median_dist, double threshold);

    double get_mean_dist_ham(std::vector<cv::DMatch> matches);

    double get_median_dist(std::vector<cv::DMatch> matches);

    std::vector<cv::DMatch> get_good_matches_of_features(const Frame& frame1, const Frame& frame2);

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
