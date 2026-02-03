#pragma once

#include <opencv2/core.hpp>

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

    // transformation matrix (in SE(3))
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

    // camera center in world/map coordinates
    cv::Mat get_position() const {
        return -R.t() * t_vec;
    }
};
