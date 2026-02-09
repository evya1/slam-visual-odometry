#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

/**
 * @class Pose
 * @brief Rigid camera pose stored as a camera->world transform T_wc (MVG2e convention mapping below).
 *
 * Stored convention (project-wide): camera -> world
 *   x_w = R_wc x_c + t_wc,   with T_wc = [R_wc | t_wc].
 *   Camera center in world coords: C_w = t_wc.
 *
 * MVG2e often writes the camera matrix as world -> camera:
 *   x ~ P X,  P = K [R_cw | t_cw].
 * Relationship:
 *   R_cw = R_wc^T
 *   t_cw = -R_wc^T t_wc
 *
 * @see geometry_conventions.h
 * @see MVG2e §9.6 (Essential matrix and pose decomposition context).
 */
class Pose {
public:
    cv::Mat R_wc; // in SO(3)
    cv::Mat t_vec; // translation (3x1)

    Pose()
        : R_wc(cv::Mat::eye(3, 3, CV_64F)),
          t_vec(cv::Mat::zeros(3, 1, CV_64F)) {
    }

    Pose(const cv::Mat &r, const cv::Mat &t)
        : R_wc(r.clone()),
          t_vec(t.clone()) {
    }

    // transformation matrix (in SE(3)) as OpenCV Mat
    cv::Mat get_transformation_matrix() const {
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        R_wc.copyTo(T(cv::Rect(0, 0, 3, 3)));
        t_vec.copyTo(T(cv::Rect(3, 0, 1, 3)));
        return T;
    }

    void set_from_transformation_matrix(const cv::Mat &T) {
        T(cv::Rect(0, 0, 3, 3)).copyTo(R_wc);
        T(cv::Rect(3, 0, 1, 3)).copyTo(t_vec);
    }

    // returns C_w
    cv::Mat get_position() const {
        return t_vec;
    }

    // == Eigen helpers ==

    // Convert stored OpenCV R to Eigen
    Eigen::Matrix3d rotation_eigen() const {
        Eigen::Matrix3d Re;
        cv::cv2eigen(R_wc, Re);
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
