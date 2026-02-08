#include "visual_odometry.h"

#include <algorithm>
#include <iostream>
#include <utility>
#include <numeric>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

VisualOdometry::VisualOdometry(int image_width, int image_height) {
    constexpr int kOrbMaxFeatures = 1200;
    constexpr float kOrbPyramidScale = 1.2f;
    constexpr int kOrbPyramidLevels = 8;
    constexpr int kOrbBorderMarginPx = 31; // edgeThreshold
    constexpr int kOrbFirstLevel = 0;
    constexpr int kOrbWtaK = 2;
    constexpr auto kOrbScoreType = cv::ORB::HARRIS_SCORE;
    constexpr int kOrbPatchSizePx = 31;
    constexpr int kOrbFastThreshold = 10;

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

void VisualOdometry::print_debugging_statistics(const double min_dist, const double max_dist,
                                                const std::size_t num_matches, const double mean_dist,
                                                const double median_dist, const double threshold) {
    std::cout << "[MatchDebug] #matches=" << num_matches
            << "  min=" << min_dist
            << "  median=" << median_dist
            << "  mean=" << mean_dist
            << "  max=" << max_dist
            << "  threshold=" << threshold
            << "  (units: Hamming bits)\n";
}

double VisualOdometry::get_mean_dist_ham(std::vector<cv::DMatch> matches) {
    double sum = 0.0;
    for (const auto &m: matches) sum += m.distance;

    const double mean_dist = sum / static_cast<double>(matches.size());
    return mean_dist;
}

double VisualOdometry::get_median_dist(std::vector<cv::DMatch> matches) {
    const std::size_t median_index = matches.size() / 2;
    const double median_dist = matches[median_index].distance;
    return median_dist;
}

std::vector<cv::DMatch> VisualOdometry::get_good_matches_of_features(const Frame &frame1, const Frame &frame2) {
    std::vector<cv::DMatch> matches;
    constexpr double kMaxHammingThreshold = 35.0;

    if (frame1.descriptors.empty() || frame2.descriptors.empty()) {
        return matches;
    }

    matcher_->match(frame1.descriptors, frame2.descriptors, matches);

    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch &a, const cv::DMatch &b) { return a.distance < b.distance; });

    if (matches.empty()) return matches;

    const double min_dist = matches.front().distance;
    const double max_dist = matches.back().distance;

    const double mean_dist = get_mean_dist_ham(matches);
    const double median_dist = get_median_dist(matches);
    const double threshold = std::min(std::max(3.0 * min_dist, 0.7 * median_dist), kMaxHammingThreshold);

    print_debugging_statistics(min_dist, max_dist, matches.size(), mean_dist, median_dist, threshold);

    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(matches.size());

    for (const auto &m: matches) {
        if (m.distance < threshold) {
            good_matches.push_back(m);
        }
    }

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
        cv::RANSAC, 0.999, 2.0, inlier_mask);

    if (E.empty()) {
        std::cerr << "Failed to compute Essential matrix\n";
        return false;
    }
    std::cout << "Essential matrix: " << E << "\n";

    // Convert E to double if needed
    cv::Mat E64;
    E.convertTo(E64, CV_64F);

    cv::Mat K64;
    camera_matrix_.convertTo(K64, CV_64F);
    cv::Mat Kinv = K64.inv();
    cv::Mat F = Kinv.t() * E64 * Kinv; // F = K^{-T} * E * K^{-1}

    std::cout << "Fundamental matrix F (pixel coords, OpenCV convention x2^T F x1 = 0):\n"
            << F << "\n";

    cv::Matx33d Fm;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Fm(r, c) = F.at<double>(r, c);

    // Build Fm from current F
    cv::Matx33d Fm_cur;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Fm_cur(r, c) = F.at<double>(r, c);

    last_F_ = Fm_cur;
    has_last_F_ = true;

    // compute epipolar residual with CURRENT F
    double mean_abs = 0.0;
    int cnt = 0;
    for (size_t i = 0; i < points1.size(); ++i) {
        if (!inlier_mask.empty() && !inlier_mask.at<uchar>((int) i)) continue;
        cv::Vec3d x1(points1[i].x, points1[i].y, 1.0);
        cv::Vec3d x2(points2[i].x, points2[i].y, 1.0);
        mean_abs += std::abs(x2.dot(Fm_cur * x1));
        cnt++;
    }
    if (cnt > 0) std::cout << "Mean |x2^T F x1| over inliers: " << (mean_abs / cnt) << "\n";

    const int inlier_count = cv::countNonZero(inlier_mask);
    std::cout << "Essential matrix computed with " << inlier_count << " inliers\n";

    const int valid_points = cv::recoverPose(E, points1, points2, camera_matrix_, R, t, inlier_mask);

    constexpr int kMinValidPoints = 10;
    constexpr int kMinInliers = 12;

    if (valid_points < kMinValidPoints || inlier_count < kMinInliers) {
        std::cerr << "Not enough valid points after pose recovery: " << valid_points
                << " (inliers=" << inlier_count << ")\n";
        return false;
    }

    std::cout << "Recovered pose with " << valid_points << " valid points\n";
    return true;
}


// Keypoint visualization
cv::Mat render_current_frame_with_keypoints_overlay(const Frame &frame) {
    cv::Mat display_image;
    cv::drawKeypoints(
        frame.image, frame.keypoints, display_image,
        cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );
    return display_image;
}

// Invert relative camera motion
void invert_relative_camera_to_camera_transform(
    const cv::Mat &R_c2_c1,
    const cv::Mat &t_c2_c1,
    cv::Mat &R_c1_c2,
    cv::Mat &t_c1_c2
) {
    // Inverse of: x_c2 = R x_c1 + t  is: x_c1 = R^T x_c2 - R^T t
    R_c1_c2 = R_c2_c1.t();
    t_c1_c2 = -R_c2_c1.t() * t_c2_c1;
}

// Compose camera->world pose
Pose compose_next_camera_to_world_pose_from_inverse_relative_motion(
    const Pose &prev_pose, // T_w_c1
    const cv::Mat &R_c1_c2, // inverse relative rotation
    const cv::Mat &t_c1_c2, // inverse relative translation (in c1 coords)
    const double scale
) {
    Pose out;
    // T_w_c2 = T_w_c1 * T_c1_c2
    out.R = prev_pose.R * R_c1_c2;
    out.t_vec = prev_pose.t_vec + scale * (prev_pose.R * t_c1_c2);
    return out;
}

void print_camera_position_debug(const Pose &pose) {
    const cv::Mat pos = pose.get_position(); // for T_w_c: should be t_vec
    std::cout << "Position: [" << pos.at<double>(0) << ", "
            << pos.at<double>(1) << ", "
            << pos.at<double>(2) << "]\n";
}


cv::Mat VisualOdometry::process_frame(Frame &frame) {
    detect_features(frame);
    const cv::Mat display_image = render_current_frame_with_keypoints_overlay(frame);

    if (!initialized_) {
        // First-frame initialization
        frame.pose = Pose(); // R = I, t = 0
        {
            std::lock_guard<std::mutex> lock(trajectory_mutex_);
            trajectory_positions_.push_back(frame.pose.get_position().clone());
            trajectory_poses_.push_back(Pose(frame.pose.R.clone(), frame.pose.t_vec.clone()));
        }
        previous_frame_ = std::move(frame);
        initialized_ = true;
        return display_image;
    }

    // Failure-safe pose default
    frame.pose = previous_frame_.pose;

    const std::vector<cv::DMatch> matches =
            get_good_matches_of_features(previous_frame_, frame);

    constexpr std::size_t kMinMatchesForPose = 10;

    if (matches.size() >= kMinMatchesForPose) {
        cv::Mat R_c2_c1, t_c2_c1;

        // NOTE: we call it once, and use BOTH the return value and the produced R,t.
        const bool pose_ok =
                estimate_relative_pose(previous_frame_, frame, matches, R_c2_c1, t_c2_c1);

        // Even if pose_ok == false (e.g. low cheirality / low valid points),
        // we still apply ROTATION-ONLY if we got a usable R,t.
        if (!R_c2_c1.empty() && !t_c2_c1.empty()) {
            constexpr double kScaleGood = 0.3;
            const double scale = pose_ok ? kScaleGood : 0.0; // rotation-only when not ok

            cv::Mat R_c1_c2, t_c1_c2;
            invert_relative_camera_to_camera_transform(
                R_c2_c1, t_c2_c1, R_c1_c2, t_c1_c2
            );

            frame.pose = compose_next_camera_to_world_pose_from_inverse_relative_motion(
                previous_frame_.pose, R_c1_c2, t_c1_c2, scale
            );

            // Optional debug:
            std::cout << "[PoseUpdate] matches=" << matches.size()
                    << " pose_ok=" << pose_ok
                    << " scale=" << scale << "\n";
        }
    }

    print_camera_position_debug(frame.pose);

    // Append trajectory every frame (even if pose didn't update)
    {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        trajectory_positions_.push_back(frame.pose.get_position().clone());
        trajectory_poses_.push_back(Pose(frame.pose.R.clone(), frame.pose.t_vec.clone()));
    }

    previous_frame_ = std::move(frame);
    return display_image;
}


std::vector<cv::Mat> VisualOdometry::get_trajectory() {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    return trajectory_positions_;
}

std::vector<Pose> VisualOdometry::get_trajectory_poses() {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    return trajectory_poses_;
}

