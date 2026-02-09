#pragma once

#include <mutex>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "frame.h"

/**
 * @class VisualOdometry
 * @brief Monocular VO using calibrated two-view geometry (MVG2e Ch. 9–10).
 *
 * Pipeline per frame i-1 -> i:
 *  1) Detect ORB features + descriptors (engineering step).
 *  2) Match descriptors (Hamming).
 *  3) Estimate Essential matrix E with RANSAC (MVG2e §9.6):
 *       x̂2^T E x̂1 = 0,  where x̂ = K^{-1} x,  E = [t]_x R.
 *  4) Decompose E to relative pose (R, t) and pick the cheirality-valid solution
 *     (MVG2e Fig. 9.12; OpenCV recoverPose does this internally).
 *  5) Convert to Fundamental matrix for pixel coords (MVG2e §9.6):
 *       F = K^{-T} E K^{-1}.
 *  6) Compose into world trajectory using project pose convention T_wc.
 *
 * Important: monocular translation scale is unobservable (MVG2e §10.2),
 * so this implementation applies a fixed visualization scale.
 *
 * @see geometry_conventions.h
 */
class VisualOdometry {
public:
    VisualOdometry(int image_width, int image_height);

    /**
     * @brief Process one frame and update its pose estimate.
     *
     * Side effects:
     *  - Detects features into frame.keypoints/descriptors.
     *  - Writes frame.pose as T_wc (camera->world).
     *  - Updates internal previous frame and stored trajectory.
     *
     * @return An OpenCV image for display (current frame with keypoint overlay).
     */
    cv::Mat process_frame(Frame &frame);

    std::vector<cv::Mat> get_trajectory();

    /**
     * @brief True iff we computed F for the most recent successful pair.
     */
    auto has_last_F() const -> bool;

    /**
     * @brief Fundamental matrix in pixel coordinates for the last pair.
     *
     * Convention (OpenCV / MVG standard):
     *   x2^T F x1 = 0  where x1 is in the previous frame, x2 in the current frame,
     *   and x = [u, v, 1]^T uses OpenCV 0-based pixels.
     *
     * @see geometry_conventions.h
     * @see MVG2e §9.2 (F) and §9.6 (relation to E).
     */
    auto last_F() const -> cv::Matx33d;


    std::vector<Pose> get_trajectory_poses();

private:
    bool has_last_F_ = false;
    cv::Matx33d last_F_{};

    void detect_features(Frame &frame);

    static void print_debugging_statistics(double min_dist, double max_dist, std::size_t num_matches, double mean_dist,
                                           double median_dist, double threshold);

    static double get_mean_dist_ham(const std::vector<cv::DMatch> &matches);

    static double get_median_dist(std::vector<cv::DMatch> matches);

    std::vector<cv::DMatch> get_good_matches_of_features(const Frame &frame1, const Frame &frame2);

    /**
     * @brief Estimate relative motion between two frames from matches using E (MVG2e §9.6).
     *
     * Inputs:
     *  - frame1 = previous frame (i-1), frame2 = current frame (i)
     *  - matches define correspondences x1 <-> x2 in pixel coords
     *
     * Outputs (OpenCV recoverPose convention):
     *  - R = R_c2_c1, t = t_c2_c1 such that (in camera coordinates)
     *      x_c2 ≈ R_c2_c1 x_c1 + t_c2_c1
     *
     * Notes:
     *  - recoverPose chooses among the 4 decompositions (MVG2e Fig. 9.12)
     *    by enforcing cheirality (positive depth in both views).
     */
    bool estimate_relative_pose(
        const Frame &frame1,
        const Frame &frame2,
        const std::vector<cv::DMatch> &matches,
        cv::Mat &R,
        cv::Mat &t
    );

    cv::Ptr<cv::ORB> orb_detector_;
    cv::Ptr<cv::BFMatcher> matcher_;
    cv::Mat camera_matrix_;

    std::vector<Pose> trajectory_poses_;
    std::vector<cv::Mat> trajectory_positions_;
    std::mutex trajectory_mutex_;

    Frame previous_frame_;
    bool initialized_ = false;
};
