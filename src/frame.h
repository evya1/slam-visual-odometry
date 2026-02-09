#pragma once
#include <vector>
#include <opencv2/core.hpp>

#include "pose.h"

/**
 * @class Frame
 * @brief One time-step in the VO pipeline: image + features + pose estimate.
 *
 * - keypoints[i].pt is a 2D pixel coordinate (OpenCV 0-based), used as
 *   x = [u, v, 1]^T in MVG notation (see geometry_conventions.h).
 * - descriptors correspond 1:1 with keypoints (ORB/Hamming).
 * - pose is the estimated camera->world pose T_wc for this frame.
 *
 * @see geometry_conventions.h
 * @see MVG2e Ch. 9–10 for two-view geometry interpretation of correspondences.
 */
class Frame {
public:
    int id;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Pose pose;
    cv::Mat image;
    double timestamp;
    bool processed;

    Frame() : id(-1), timestamp(0.0), processed(false) {
    }

    Frame(int frame_id, const cv::Mat &img)
        : id(frame_id), image(img.clone()), timestamp(0.0), processed(false) {
    }

    Frame(Frame &&other) noexcept
        : id(other.id),
          keypoints(std::move(other.keypoints)),
          descriptors(std::move(other.descriptors)),
          pose(std::move(other.pose)),
          image(std::move(other.image)),
          timestamp(other.timestamp),
          processed(other.processed) {
    }

    Frame &operator=(Frame &&other) noexcept {
        if (this != &other) {
            id = other.id;
            keypoints = std::move(other.keypoints);
            descriptors = std::move(other.descriptors);
            pose = std::move(other.pose);
            image = std::move(other.image);
            timestamp = other.timestamp;
            processed = other.processed;
        }
        return *this;
    }
};