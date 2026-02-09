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
    int id; // Unique frame identifier
    std::vector<cv::KeyPoint> keypoints; // Detected feature points
    cv::Mat descriptors; // Feature descriptors
    Pose pose; // Camera pose at this frame
    cv::Mat image; // The image itself (optional)
    double timestamp; // Capture time (optional)
    bool processed; // Whether frame was processed

    Frame() : id(-1), timestamp(0.0), processed(false) {
    }

    Frame(int frame_id, const cv::Mat &img)
        : id(frame_id), image(img.clone()), timestamp(0.0), processed(false) {
    }

    // Move constructor
    Frame(Frame &&other) noexcept
        : id(other.id),
          keypoints(std::move(other.keypoints)),
          descriptors(std::move(other.descriptors)),
          pose(std::move(other.pose)),
          image(std::move(other.image)),
          timestamp(other.timestamp),
          processed(other.processed) {
    }

    // Move assignment operator
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