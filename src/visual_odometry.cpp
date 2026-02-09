#include "visual_odometry.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace {
    // Distance (in pixels) from point x=(u,v) to line l=(a,b,c) with a*u + b*v + c = 0.
    double pointLineDistancePx(const cv::Vec3d &l, const cv::Point2f &x) {
        const double a = l[0], b = l[1], c = l[2];
        const double num = std::abs(a * x.x + b * x.y + c);
        const double den = std::sqrt(a * a + b * b);
        return (den > 1e-12) ? (num / den) : std::numeric_limits<double>::infinity();
    }

    // Sanity-check: for inlier matches (x1_i, x2_i), compute dist(x2_i, l2_i) where l2_i = F * x1_i.
    // If F and ordering are correct, this should be small (often ~1–3 px, depending on noise/thresholds).
    void debugEpipolarResidualsPx(const cv::Matx33d &F,
                                  const std::vector<cv::Point2f> &points1,
                                  const std::vector<cv::Point2f> &points2,
                                  const cv::Mat &inlier_mask,
                                  int maxPrint = 10) {
        int printed = 0;
        double sum = 0.0, maxd = 0.0;
        int cnt = 0;

        const bool hasMask = !inlier_mask.empty() && inlier_mask.type() == CV_8U;

        const int N = static_cast<int>(std::min(points1.size(), points2.size()));
        for (int i = 0; i < N; ++i) {
            if (hasMask && inlier_mask.at<uchar>(i) == 0) continue;

            const cv::Vec3d x1(points1[i].x, points1[i].y, 1.0);
            const cv::Vec3d l2 = F * x1; // l2 = F x1
            const double d = pointLineDistancePx(l2, points2[i]); // dist(x2, l2)

            sum += d;
            maxd = std::max(maxd, d);
            ++cnt;

            if (printed < maxPrint) {
                std::cout << "[epi] inlier " << i << "  d(x2, F x1) = " << d << " px\n";
                ++printed;
            }
        }

        if (cnt > 0) {
            std::cout << "[epi] inlier residuals: mean=" << (sum / cnt)
                    << " px, max=" << maxd << " px, N=" << cnt << "\n";
        } else {
            std::cout << "[epi] no inliers to check.\n";
        }
    }
} // namespace

VisualOdometry::VisualOdometry(int image_width, int image_height) {
    constexpr int kOrbMaxFeatures = 1200;
    constexpr float kOrbPyramidScale = 1.2f;
    constexpr int kOrbPyramidLevels = 8;
    constexpr int kOrbBorderMarginPx = 31; // ORB edgeThreshold
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

    // Intrinsics (K) used for calibrated two-view geometry.
    const auto fx = static_cast<double>(image_width); // focal length in pixels
    const auto fy = static_cast<double>(image_width); // assume square pixels
    const auto cx = image_width / 2.0;
    const auto cy = image_height / 2.0;

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

auto VisualOdometry::get_mean_dist_ham(const std::vector<cv::DMatch> &matches) -> double {
    double sum = 0.0;
    for (const auto &m: matches) sum += m.distance;
    return sum / static_cast<double>(matches.size());
}

auto VisualOdometry::get_median_dist(std::vector<cv::DMatch> matches) -> double {
    // Median of distances without fully sorting.
    const std::size_t mid = matches.size() / 2;
    std::nth_element(matches.begin(), matches.begin() + static_cast<long>(mid), matches.end(),
                     [](const cv::DMatch &a, const cv::DMatch &b) { return a.distance < b.distance; });
    return matches[mid].distance;
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
        points1.push_back(frame1.keypoints[m.queryIdx].pt); // prev frame
        points2.push_back(frame2.keypoints[m.trainIdx].pt); // curr frame
    }

    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(
        points1, points2, camera_matrix_,
        cv::RANSAC, 0.999, 2.0, inlier_mask
    );

    if (E.empty()) {
        std::cerr << "Failed to compute Essential matrix\n";
        return false;
    }
    std::cout << "Essential matrix:\n" << E << "\n";

    // Compute F in pixel coordinates: F = K^{-T} E K^{-1}.
    cv::Mat E64;
    E.convertTo(E64, CV_64F);

    cv::Mat K64;
    camera_matrix_.convertTo(K64, CV_64F);
    const cv::Mat Kinv = K64.inv();

    const cv::Mat F = Kinv.t() * E64 * Kinv;
    std::cout << "Fundamental matrix F (pixel coords, OpenCV convention x2^T F x1 = 0):\n"
            << F << "\n";

    auto mat3x3_to_matx33d = [](const cv::Mat &M) -> cv::Matx33d {
        cv::Matx33d out;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                out(r, c) = M.at<double>(r, c);
        return out;
    };

    const cv::Matx33d F_px = mat3x3_to_matx33d(F);
    last_F_ = F_px;
    has_last_F_ = true;

    // Algebraic epipolar residual (unnormalized): |x2^T F x1| over inliers.
    double mean_abs = 0.0;
    int cnt = 0;
    for (size_t i = 0; i < points1.size(); ++i) {
        if (!inlier_mask.empty() && inlier_mask.type() == CV_8U &&
            !inlier_mask.at<uchar>(static_cast<int>(i))) {
            continue;
        }
        const cv::Vec3d x1(points1[i].x, points1[i].y, 1.0);
        const cv::Vec3d x2(points2[i].x, points2[i].y, 1.0);
        mean_abs += std::abs(x2.dot(F_px * x1));
        ++cnt;
    }
    if (cnt > 0) {
        std::cout << "Mean |x2^T F x1| over inliers: " << (mean_abs / cnt) << "\n";
    }

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

cv::Mat render_current_frame_with_keypoints_overlay(const Frame &frame) {
    cv::Mat display_image;
    cv::drawKeypoints(
        frame.image, frame.keypoints, display_image,
        cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );
    return display_image;
}

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

Pose compose_next_camera_to_world_pose_from_inverse_relative_motion(
    const Pose &prev_pose, // T_w_c1
    const cv::Mat &R_c1_c2, // inverse relative rotation
    const cv::Mat &t_c1_c2, // inverse relative translation (in c1 coords)
    const double scale
) {
    Pose out;
    // T_w_c2 = T_w_c1 * T_c1_c2
    out.R_wc = prev_pose.R_wc * R_c1_c2;
    out.t_wc = prev_pose.t_wc + scale * (prev_pose.R_wc * t_c1_c2);
    return out;
}

void print_camera_position_debug(const Pose &pose) {
    const cv::Mat& pos = pose.C_w();
    std::cout << "Position: [" << pos.at<double>(0) << ", "
            << pos.at<double>(1) << ", "
            << pos.at<double>(2) << "]\n";
}

cv::Mat VisualOdometry::process_frame(Frame &frame) {
    detect_features(frame);
    const cv::Mat display_image = render_current_frame_with_keypoints_overlay(frame);

    if (!initialized_) {
        frame.pose = Pose(); // R_wc = I, t_wc = 0
        {
            std::lock_guard<std::mutex> lock(trajectory_mutex_);
            trajectory_positions_.push_back(frame.pose.C_w().clone());
            trajectory_poses_.emplace_back(frame.pose.R_wc.clone(), frame.pose.t_wc.clone());
        }
        previous_frame_ = std::move(frame);
        initialized_ = true;
        return display_image;
    }

    // Default to last known pose if the current update fails.
    frame.pose = previous_frame_.pose;

    const std::vector<cv::DMatch> matches = get_good_matches_of_features(previous_frame_, frame);

    constexpr std::size_t kMinMatchesForPose = 10;
    if (matches.size() >= kMinMatchesForPose) {
        cv::Mat R_c2_c1, t_c2_c1;

        const bool pose_ok = estimate_relative_pose(previous_frame_, frame, matches, R_c2_c1, t_c2_c1);

        // If pose_ok is false, we still apply rotation-only (translation set to 0).
        if (!R_c2_c1.empty() && !t_c2_c1.empty()) {
            constexpr double kScaleGood = 0.3;
            const double scale = pose_ok ? kScaleGood : 0.0;

            cv::Mat R_c1_c2, t_c1_c2;
            invert_relative_camera_to_camera_transform(R_c2_c1, t_c2_c1, R_c1_c2, t_c1_c2);

            frame.pose = compose_next_camera_to_world_pose_from_inverse_relative_motion(
                previous_frame_.pose, R_c1_c2, t_c1_c2, scale
            );

            std::cout << "[PoseUpdate] matches=" << matches.size()
                    << " pose_ok=" << pose_ok
                    << " scale=" << scale << "\n";
        }
    }

    print_camera_position_debug(frame.pose);

    {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        trajectory_positions_.push_back(frame.pose.C_w().clone());
        trajectory_poses_.emplace_back(frame.pose.R_wc.clone(), frame.pose.t_wc.clone());
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

auto VisualOdometry::has_last_F() const -> bool { return has_last_F_; }
auto VisualOdometry::last_F() const -> cv::Matx33d { return last_F_; }
