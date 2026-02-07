#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

class Pose {
public:
    cv::Mat R;      // R in SO(3)
    cv::Mat t_vec;  // translation (3x1)

    Pose()
        : R(cv::Mat::eye(3, 3, CV_64F)),
          t_vec(cv::Mat::zeros(3, 1, CV_64F)) {}

    Pose(const cv::Mat& r, const cv::Mat& t)
        : R(r.clone()),
          t_vec(t.clone()) {}

    // transformation matrix (in SE(3)) as OpenCV Mat
    cv::Mat get_transformation_matrix() const {
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        R.copyTo(T(cv::Rect(0, 0, 3, 3)));
        t_vec.copyTo(T(cv::Rect(3, 0, 1, 3)));
        return T;
    }

    void set_from_transformation_matrix(const cv::Mat& T) {
        T(cv::Rect(0, 0, 3, 3)).copyTo(R);
        T(cv::Rect(3, 0, 1, 3)).copyTo(t_vec);
    }

    // camera center in world/map coordinates (for world->camera convention)
    cv::Mat get_position() const {
        return -R.t() * t_vec;
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

    // Treat Pose as world->camera:  x_c = R_cw x_w + t_cw
    Eigen::Matrix4d T_c_w_eigen() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.topLeftCorner<3, 3>() = rotation_eigen();
        T.topRightCorner<3, 1>() = translation_eigen();
        return T;
    }

    // Inverse transform camera->world (useful for Pangolin drawing)
    Eigen::Matrix4d T_w_c_eigen() const {
        const Eigen::Matrix3d Rcw = rotation_eigen();
        const Eigen::Vector3d tcw = translation_eigen();

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.topLeftCorner<3, 3>() = Rcw.transpose();            // Rwc
        T.topRightCorner<3, 1>() = -Rcw.transpose() * tcw;    // twc
        return T;
    }

    // Camera center in world coordinates (Eigen)
    Eigen::Vector3d position_w_eigen() const {
        const Eigen::Matrix3d Rcw = rotation_eigen();
        const Eigen::Vector3d tcw = translation_eigen();
        return -Rcw.transpose() * tcw;
    }
};
