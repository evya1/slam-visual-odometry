#include "visual_odometry.h"

#include <algorithm>
#include <iostream>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

VisualOdometry::VisualOdometry(int image_width, int image_height) {
    constexpr int   kOrbMaxFeatures     = 300;
    constexpr float kOrbPyramidScale    = 1.2f;
    constexpr int   kOrbPyramidLevels   = 8;
    constexpr int   kOrbBorderMarginPx  = 31;  // edgeThreshold
    constexpr int   kOrbFirstLevel      = 0;
    constexpr int   kOrbWtaK            = 2;
    constexpr auto  kOrbScoreType       = cv::ORB::HARRIS_SCORE;
    constexpr int   kOrbPatchSizePx     = 31;
    constexpr int   kOrbFastThreshold   = 20;

    orb_detector_ = cv::ORB::create(
        kOrbMaxFeatures,
        kOrbPyramidScale,
        kOrbPyramidLevels,
        kOrbBorderMarginPx,
        kOrbFirstLevel,
        kOrbWtaK,
        kOrbScoreType,
        kOrbPatchSizePx,
        kOrbFastThreshold
    );

    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);

    // Construct simple camera intrinsic matrix from image dimensions
    // Assuming principal point at image center and focal length ~ image width
    const double fx = image_width; // focal length in pixels
    const double fy = image_width; // assuming square pixels
    const double cx = image_width / 2.0;
    const double cy = image_height / 2.0;

    camera_matrix_ = (cv::Mat_<double>(3, 3) <<
                      fx, 0, cx,
                      0, fy, cy,
                      0, 0, 1);

    std::cout << "Camera matrix initialized:\n" << camera_matrix_ << std::endl;
}

void VisualOdometry::detect_features(Frame &frame) {
    cv::Mat gray;
    const int channels = frame.image.channels();

    if (channels == 3) {
        cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.image;
    }

    orb_detector_->detectAndCompute(gray, cv::noArray(), frame.keypoints, frame.descriptors);
    frame.processed = true;

    std::cout << "Frame " << frame.id << ": Detected " << frame.keypoints.size() << " keypoints\n";
}

std::vector<cv::DMatch> VisualOdometry::match_features(const Frame &frame1, const Frame &frame2) {
    std::vector<cv::DMatch> matches;

    if (frame1.descriptors.empty() || frame2.descriptors.empty()) {
        return matches;
    }

    matcher_->match(frame1.descriptors, frame2.descriptors, matches);

    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch &a, const cv::DMatch &b) { return a.distance < b.distance; });
    // --- DEBUG: distance stats ---
    if (!matches.empty()) {
        const double min_d = matches.front().distance;
        const double max_d = matches.back().distance;

        double sum = 0.0;
        for (const auto &m: matches) sum += m.distance;
        const double mean_d = sum / matches.size();

        const double median_d = matches[matches.size() / 2].distance;

        const double threshold = std::min(1.5 * min_d, 25.0);

        std::cout << "[MatchDebug] #matches=" << matches.size()
                << "  min=" << min_d
                << "  median=" << median_d
                << "  mean=" << mean_d
                << "  max=" << max_d
                << "  threshold=" << threshold
                << "  (units: Hamming bits)\n";
    }

    std::vector<cv::DMatch> good_matches;
    const double min_dist = matches.empty() ? 0.0 : matches.front().distance;
    const double threshold = std::min(1.5 * min_dist, 25.0);

    for (const auto &m: matches) {
        if (m.distance < threshold) {
            good_matches.push_back(m);
        }
    }

    std::cout << "Matched " << good_matches.size() << " features (from " << matches.size() << " total)\n";
    return good_matches;
}

bool VisualOdometry::estimate_relative_pose(
    const Frame &frame1,
    const Frame &frame2,
    const std::vector<cv::DMatch> &matches,
    cv::Mat &R,
    cv::Mat &t
) {
    if (matches.size() < 8) {
        std::cerr << "Not enough matches for pose estimation\n";
        return false;
    }

    std::vector<cv::Point2f> points1, points2;
    points1.reserve(matches.size());
    points2.reserve(matches.size());

    for (const auto &m: matches) {
        if (m.queryIdx < 0 ||
            m.queryIdx >= static_cast<int>(frame1.keypoints.size()) ||
            m.trainIdx < 0 ||
            m.trainIdx >= static_cast<int>(frame2.keypoints.size())) {
            std::cerr << "Skipping match with out-of-range indices: "
                    << "queryIdx=" << m.queryIdx
                    << ", trainIdx=" << m.trainIdx << "\n";
            continue;
        }
        points1.push_back(frame1.keypoints[m.queryIdx].pt);
        points2.push_back(frame2.keypoints[m.trainIdx].pt);
    }

    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(
        points1, points2, camera_matrix_,
        cv::RANSAC, 0.999, 1.0, inlier_mask
    );

    if (E.empty()) {
        std::cerr << "Failed to compute Essential matrix\n";
        return false;
    }

    const int inlier_count = cv::countNonZero(inlier_mask);
    std::cout << "Essential matrix computed with " << inlier_count << " inliers\n";

    const int valid_points = cv::recoverPose(E, points1, points2, camera_matrix_, R, t, inlier_mask);

    if (valid_points < 10) {
        std::cerr << "Not enough valid points after pose recovery: " << valid_points << "\n";
        return false;
    }

    std::cout << "Recovered pose with " << valid_points << " valid points\n";
    return true;
}

cv::Mat VisualOdometry::process_frame(Frame &frame) {
    detect_features(frame);

    cv::Mat display_image;
    cv::drawKeypoints(
        frame.image, frame.keypoints, display_image,
        cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    if (!initialized_) {
        frame.pose = Pose();
        {
            std::lock_guard<std::mutex> lock(trajectory_mutex_);
            trajectory_positions_.push_back(frame.pose.get_position().clone());
        }
        previous_frame_ = std::move(frame);
        initialized_ = true;
        return display_image;
    }

    const std::vector<cv::DMatch> matches = match_features(previous_frame_, frame);

    if (matches.size() >= 8) {
        cv::Mat R, t;
        if (estimate_relative_pose(previous_frame_, frame, matches, R, t)) {
            const cv::Mat R_prev = previous_frame_.pose.R;
            const cv::Mat t_prev = previous_frame_.pose.t_vec;

            // Scale factor:
            //   In a monocular visual odometry system, absolute scale cannot be
            //   recovered from images alone. In a real system, the scale would
            //   come from additional sensors (e.g., IMU, wheel odometry, depth)
            //   or prior knowledge about the scene/trajectory.
            //   For this demo, use a fixed scale.
            const double scale = 0.1;

            frame.pose.R = R_prev * R;
            frame.pose.t_vec = t_prev + scale * R_prev * t;

            {
                std::lock_guard<std::mutex> lock(trajectory_mutex_);
                trajectory_positions_.push_back(frame.pose.get_position().clone());
            }

            const cv::Mat pos = frame.pose.get_position();
            std::cout << "Position: [" << pos.at<double>(0) << ", "
                                    << pos.at<double>(1) << ", "
                                    << pos.at<double>(2) << "]\n";
        }
    }

    previous_frame_ = std::move(frame);
    return display_image;
}

std::vector<cv::Mat> VisualOdometry::get_trajectory() {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    return trajectory_positions_;
}
