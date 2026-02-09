#pragma once

#include <opencv2/core.hpp>

/**
 * @class Pose
 * @brief Rigid camera pose stored as a camera->world transform T_wc.
 *
 * Stored convention (project-wide): {C} frame -> {W} frame.
 * Meaning `R_wc` maps a vector’s coordinates expressed in frame {C} into frame {W}.
 * where R_cw (i.e., R_wc^T) is change-of-coordinates matrix from {W} -> {C}.
 *
 *   x_w = R_wc x_c + t_wc,   with T_wc = [R_wc | t_wc].
 *   Camera center in world coords: C_w = t_wc.
 *
 * MVG2e often writes the camera matrix as world -> camera:
 *   x ~ P X,  P = K [R_cw | t_cw].
 *
 * Relationship:
 *   R_cw = R_wc^T
 *   t_cw = -R_wc^T t_wc
 *
 * @see geometry_conventions.h
 * @see MVG2e §9.6 (Essential matrix and pose decomposition context).
 */
class Pose {
public:
    // Stored convention: camera -> world
    //   X_w = R_wc * X_c + t_wc
    //   C_w = t_wc
    cv::Mat R_wc; // 3x3 CV_64F
    cv::Mat t_wc; // 3x1 CV_64F

    Pose()
        : R_wc(cv::Mat::eye(3, 3, CV_64F)),
          t_wc(cv::Mat::zeros(3, 1, CV_64F)) {
    }

    Pose(const cv::Mat &R, const cv::Mat &t) {
        set_R_t(R, t);
    }

    void set_R_t(const cv::Mat &R, const cv::Mat &t) {
        CV_Assert(R.rows == 3 && R.cols == 3);
        CV_Assert((t.rows == 3 && t.cols == 1) || (t.rows == 1 && t.cols == 3));

        R.convertTo(R_wc, CV_64F);

        cv::Mat t_col = (t.rows == 3) ? t : t.t(); // ensuring 3x1
        t_col.convertTo(t_wc, CV_64F);
    }

    cv::Mat T_wc() const {
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        R_wc.copyTo(T(cv::Rect(0, 0, 3, 3)));
        t_wc.copyTo(T(cv::Rect(3, 0, 1, 3)));
        return T;
    }

    // World -> camera (MVG extrinsic form)
    cv::Mat R_cw() const { return R_wc.t(); }

    cv::Mat t_cw() const {
        cv::Mat Rcw = R_wc.t();
        return -Rcw * t_wc;
    }

    cv::Mat T_cw() const {
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        cv::Mat Rcw = R_cw();
        cv::Mat tcw = t_cw();
        Rcw.copyTo(T(cv::Rect(0, 0, 3, 3)));
        tcw.copyTo(T(cv::Rect(3, 0, 1, 3)));
        return T;
    }

    const cv::Mat &C_w() const { return t_wc; } // camera centre in world coordinates
};
