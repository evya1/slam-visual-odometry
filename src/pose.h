#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

// Pose convention: camera -> world (T_w_c)
//
// We store a rigid transform that maps coordinates from the camera frame to the world frame:
//
//   x_w = R_wc * x_c + t_wc
//
// where R_wc ∈ SO(3) and t_wc ∈ R^3.
//
// Homogeneous form:
//
//   T_wc = [ R_wc  t_wc ]
//          [  0      1  ]
//
// Camera center (camera position in world coordinates):
// The camera center is the world coordinate of the camera-frame origin (x_c = 0):
//
//   C_w = t_wc
//
// (If using the opposite convention world -> camera: x_c = R_cw x_w + t_cw,
//  then R_cw = R_wc^T, t_cw = -R_wc^T t_wc, and C_w = -R_cw^T t_cw.)
class Pose {
public:
    cv::Mat R; // R in SO(3)
    cv::Mat t_vec; // translation (3x1)

    Pose()
        : R(cv::Mat::eye(3, 3, CV_64F)),
          t_vec(cv::Mat::zeros(3, 1, CV_64F)) {
    }

    Pose(const cv::Mat &r, const cv::Mat &t)
        : R(r.clone()),
          t_vec(t.clone()) {
    }

    // transformation matrix (in SE(3)) as OpenCV Mat
    cv::Mat get_transformation_matrix() const {
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        R.copyTo(T(cv::Rect(0, 0, 3, 3)));
        t_vec.copyTo(T(cv::Rect(3, 0, 1, 3)));
        return T;
    }

    void set_from_transformation_matrix(const cv::Mat &T) {
        T(cv::Rect(0, 0, 3, 3)).copyTo(R);
        T(cv::Rect(3, 0, 1, 3)).copyTo(t_vec);
    }

    // Camera center in world coordinates.
    //
    // Pose convention here is camera->world (T_w_c):
    //   $x_w = R_{wc}\,x_c + t_{wc}$
    // so the camera center is:
    //   $C_w = t_{wc}$
    //
    // @return cv::Mat 3x1 (CV_64F) column vector representing $C_w$.
    cv::Mat get_position() const {
        return t_vec;
    }

    // == Eigen helpers ==

    // Convert stored OpenCV R to Eigen
    Eigen::Matrix3d rotation_eigen() const {
        Eigen::Matrix3d Re;
        cv::cv2eigen(R, Re);
        return Re;
    }

    // Convert stored OpenCV t to Eigen
    Eigen::Vector3d translation_eigen() const {
        Eigen::Vector3d te;
        cv::cv2eigen(t_vec, te);
        return te;
    }

    // Inverse transform world->camera (useful sometimes)
    Eigen::Matrix4d T_c_w_eigen() const {
        const Eigen::Matrix3d Rwc = rotation_eigen();
        const Eigen::Vector3d twc = translation_eigen();

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.topLeftCorner<3, 3>() = Rwc.transpose(); // Rcw
        T.topRightCorner<3, 1>() = -Rwc.transpose() * twc; // tcw
        return T;
    }

    // Camera center in world coordinates (Eigen)
    Eigen::Vector3d position_w_eigen() const {
        return translation_eigen(); // $C_w = t_{wc}$
    }
};
