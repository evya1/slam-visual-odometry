/**
 * main.cpp - Epipolar Visual Odometry
 *
 * A Visual Odometry system that processes a sequence of images captured by a drone,
 * estimates the camera motion between consecutive frames (3D Rotation + Translation),
 * and visualizes the result in real time using OpenCV and Pangolin.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <mutex>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <pangolin/pangolin.h>

namespace fs = std::filesystem;
class Pose {
public:
    cv::Mat R;     // R in SO(3)
    cv::Mat t_vec;

    Pose() {
        R = cv::Mat::eye(3, 3, CV_64F);
        t_vec = cv::Mat::zeros(3, 1, CV_64F);
    }

    Pose(const cv::Mat& r, const cv::Mat& t) {
        R = r.clone();
        t_vec = t.clone();
    }

    // getter and setter for transformation matrix (in SE(3))
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

    cv::Mat get_position() const {
        return -R.t() * t_vec;
    }
};

// =============================================================================
// Frame Class - Represents a single image in the sequence
// =============================================================================
class Frame {
public:
    int id;                                  // Unique frame identifier
    std::vector<cv::KeyPoint> keypoints;     // Detected feature points
    cv::Mat descriptors;                     // Feature descriptors
    Pose pose;                               // Camera pose at this frame
    cv::Mat image;                           // The image itself (optional)
    double timestamp;                        // Capture time (optional)
    bool processed;                          // Whether frame was processed

    Frame() : id(-1), timestamp(0.0), processed(false) {}

    Frame(int frame_id, const cv::Mat& img)
        : id(frame_id), image(img.clone()), timestamp(0.0), processed(false) {}
};

// =============================================================================
// Visual Odometry Class
// =============================================================================
class VisualOdometry {
private:
    cv::Ptr<cv::ORB> orb_detector;
    cv::Ptr<cv::BFMatcher> matcher;
    cv::Mat camera_matrix;

    std::vector<cv::Mat> trajectory_positions;  // Store all camera positions
    std::mutex trajectory_mutex;

    Frame previous_frame;
    bool initialized;

public:
    VisualOdometry(int image_width, int image_height) : initialized(false) {
        const int nfeatures = 2000;
        // Initialize ORB detector with reasonable parameters
        orb_detector = cv::ORB::create(nfeatures, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

        // Initialize brute-force matcher with Hamming distance for ORB
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);

        // Construct simple camera intrinsic matrix from image dimensions
        // Assuming principal point at image center and focal length ~ image width
        double fx = image_width;   // focal length in pixels
        double fy = image_width;   // assuming square pixels
        double cx = image_width / 2.0;
        double cy = image_height / 2.0;

        camera_matrix = (cv::Mat_<double>(3, 3) <<
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1);

        std::cout << "Camera matrix initialized:" << std::endl << camera_matrix << std::endl;
    }

    // Detect keypoints and compute descriptors for a frame
    void detect_features(Frame& frame) {
        cv::Mat gray;
        int channels = frame.image.channels();

        if (channels == 3) {
            cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame.image;
        }

        orb_detector->detectAndCompute(gray, cv::noArray(), frame.keypoints, frame.descriptors);
        frame.processed = true;

        std::cout << "Frame " << frame.id << ": Detected " << frame.keypoints.size() << " keypoints" << std::endl;
    }

    // Match features between two frames
    std::vector<cv::DMatch> match_features(const Frame& frame1, const Frame& frame2) {
        std::vector<cv::DMatch> matches;

        if (frame1.descriptors.empty() || frame2.descriptors.empty()) {
            return matches;
        }

        matcher->match(frame1.descriptors, frame2.descriptors, matches);

        // Sort matches by distance
        std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

        // Keep only good matches (ratio test)
        std::vector<cv::DMatch> good_matches;
        double min_dist = matches.empty() ? 0 : matches[0].distance;
        double threshold = std::max(2.0 * min_dist, 30.0);

        for (const auto& m : matches) {
            if (m.distance < threshold) {
                good_matches.push_back(m);
            }
        }

        std::cout << "Matched " << good_matches.size() << " features (from " << matches.size() << " total)" << std::endl;

        return good_matches;
    }

    // Estimate relative pose between two frames using Essential matrix
    bool estimate_relative_pose(const Frame& frame1, const Frame& frame2,
                                const std::vector<cv::DMatch>& matches,
                                cv::Mat& R, cv::Mat& t) {
        if (matches.size() < 8) {
            std::cerr << "Not enough matches for pose estimation" << std::endl;
            return false;
        }

        // Extract matched points
        std::vector<cv::Point2f> points1, points2;
        for (const auto& m : matches) {
            points1.push_back(frame1.keypoints[m.queryIdx].pt);
            points2.push_back(frame2.keypoints[m.trainIdx].pt);
        }

        // Compute Essential matrix using RANSAC
        cv::Mat inlier_mask;
        cv::Mat E = cv::findEssentialMat(points1, points2, camera_matrix,
                                          cv::RANSAC, 0.999, 1.0, inlier_mask);

        if (E.empty()) {
            std::cerr << "Failed to compute Essential matrix" << std::endl;
            return false;
        }

        // Count inliers
        int inlier_count = cv::countNonZero(inlier_mask);
        std::cout << "Essential matrix computed with " << inlier_count << " inliers" << std::endl;

        // Recover pose from Essential matrix
        int valid_points = cv::recoverPose(E, points1, points2, camera_matrix, R, t, inlier_mask);

        if (valid_points < 10) {
            std::cerr << "Not enough valid points after pose recovery: " << valid_points << std::endl;
            return false;
        }

        std::cout << "Recovered pose with " << valid_points << " valid points" << std::endl;

        return true;
    }

    // Process a new frame and update trajectory
    cv::Mat process_frame(Frame& frame) {
        detect_features(frame);

        cv::Mat display_image;
        cv::drawKeypoints(frame.image, frame.keypoints, display_image,
                          cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        if (!initialized) {
            // First frame - initialize with identity pose
            frame.pose = Pose();
            {
                std::lock_guard<std::mutex> lock(trajectory_mutex);
                trajectory_positions.push_back(frame.pose.get_position().clone());
            }
            previous_frame = frame;
            initialized = true;
            return display_image;
        }

        // Match with previous frame
        std::vector<cv::DMatch> matches = match_features(previous_frame, frame);

        if (matches.size() >= 8) {
            cv::Mat R, t;
            if (estimate_relative_pose(previous_frame, frame, matches, R, t)) {
                // Accumulate pose: T_world_current = T_world_previous * T_previous_current
                cv::Mat R_prev = previous_frame.pose.R;
                cv::Mat t_prev = previous_frame.pose.t_vec;

                // Scale factor (in real system, would come from additional sensors or prior knowledge)
                // For this demo, use a fixed scale
                double scale = 0.1;

                // Update rotation & translation
                frame.pose.R    = R_prev * R;
                frame.pose.t_vec = t_prev + scale * R * t;

                // Store position for trajectory
                {
                    std::lock_guard<std::mutex> lock(trajectory_mutex);
                    trajectory_positions.push_back(frame.pose.get_position().clone());
                }

                cv::Mat pos = frame.pose.get_position();
                std::cout << "Position: [" << pos.at<double>(0) << ", "
                          << pos.at<double>(1) << ", " << pos.at<double>(2) << "]" << std::endl;
            }
        }

        previous_frame = frame;
        return display_image;
    }

    // Get copy of trajectory positions (thread-safe)
    std::vector<cv::Mat> get_trajectory() {
        std::lock_guard<std::mutex> lock(trajectory_mutex);
        return trajectory_positions;
    }
};

// =============================================================================
// Trajectory Viewer Class (Pangolin-based)
// =============================================================================
class TrajectoryViewer {
public:
    void init() {
        if (initialized) return;

        pangolin::CreateWindowAndBind("Visual Odometry: Trajectory", 1024, 768);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(
                0, -1.5, -3.0,          // closer eye position
                0, 0.0, 0.0,          // look at origin
                pangolin::AxisNegY
            )
        );

        d_cam = &pangolin::CreateDisplay()
                    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                    .SetHandler(new pangolin::Handler3D(s_cam));

        initialized = true;
    }

    bool should_quit() const {
        return pangolin::ShouldQuit();
    }

    void render_step(const std::vector<cv::Mat>& trajectory) {
        init();
        // Center the initial view on the current trajectory (run once)
        if (!home_set && !trajectory.empty()) {
            const auto& p = trajectory.back();
            const double cx = p.at<double>(0);
            const double cy = p.at<double>(1);
            const double cz = p.at<double>(2);

            const double yaw_deg = 45.0;
            const double yaw = yaw_deg * M_PI / 180.0;

            // original offset from target
            const double dx = 0.0;
            const double dy = -1.5;
            const double dz = -3.0;

            // yaw rotation around vertical axis (rotate in x-z)
            const double dx_rot = dx * std::cos(yaw) + dz * std::sin(yaw);
            const double dz_rot = -dx * std::sin(yaw) + dz * std::cos(yaw);

            s_cam.SetModelViewMatrix(
                pangolin::ModelViewLookAt(
                    cx + dx_rot, cy + dy, cz + dz_rot,      // eye
                    cx,          cy,      cz,            // target
                    pangolin::AxisNegY
                )
            );
            home_set = true;
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

        d_cam->Activate(s_cam);
        const float axis_display_length = 1.0f;
        draw_axes(axis_display_length);
        draw_grid();

        if (trajectory.size() > 1) {
            glColor3f(0.0f, 1.0f, 0.0f);
            glLineWidth(2.0f);
            glBegin(GL_LINE_STRIP);
            for (const auto& pos : trajectory) {
                glVertex3d(pos.at<double>(0), pos.at<double>(1), pos.at<double>(2));
            }
            glEnd();

            glPointSize(5.0f);
            glBegin(GL_POINTS);
            for (size_t i = 0; i < trajectory.size(); ++i) {
                const auto& pos = trajectory[i];
                if (i == 0) glColor3f(1.0f, 0.0f, 0.0f);
                else if (i == trajectory.size() - 1) glColor3f(0.0f, 0.0f, 1.0f);
                else glColor3f(0.0f, 1.0f, 0.0f);
                glVertex3d(pos.at<double>(0), pos.at<double>(1), pos.at<double>(2));
            }
            glEnd();
        }

        pangolin::FinishFrame(); // processes events too
    }

private:
    bool initialized = false;
    bool home_set = false;
    pangolin::OpenGlRenderState s_cam;
    pangolin::View* d_cam = nullptr;

    void draw_axes(const float axis_display_length) {
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(0, 0, 0); glVertex3f(axis_display_length, 0, 0);  // +X (red)
        glColor3f(0.0f, 1.0f, 0.0f); glVertex3f(0, 0, 0); glVertex3f(0, axis_display_length, 0);  // +Y (green)
        glColor3f(0.0f, 0.0f, 1.0f); glVertex3f(0, 0, 0); glVertex3f(0, 0, axis_display_length);  // +Z (blue)
        glEnd();
    }

    void draw_grid() {
        glColor3f(0.3f, 0.3f, 0.3f);
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        float grid_size = 10.0f;
        float step = 1.0f;
        for (float i = -grid_size; i <= grid_size; i += step) {
            glVertex3f(-grid_size, 0, i); glVertex3f(grid_size, 0, i);
            glVertex3f(i, 0, -grid_size); glVertex3f(i, 0, grid_size);
        }
        glEnd();
    }
};
// =============================================================================
// Utility Functions
// =============================================================================

// Load image paths from dataset directory (sorted lexicographically)
std::vector<std::string> load_image_paths(const std::string& dataset_path) {
    std::vector<std::string> image_paths;

    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            std::string ext = entry.path().extension().string();

            // Convert extension to lowercase for comparison
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                image_paths.push_back(path);
            }
        }
    }

    // Sort lexicographically
    std::sort(image_paths.begin(), image_paths.end());

    std::cout << "Found " << image_paths.size() << " images in dataset" << std::endl;

    return image_paths;
}

// =============================================================================
// Main Function
// =============================================================================
int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Epipolar Visual Odometry - NAV HW2   " << std::endl;
    std::cout << "========================================" << std::endl;

    // Dataset path: prefer CLI arg, otherwise auto-detect common locations.
    std::string dataset_path;

    if (argc > 1) {
        dataset_path = argv[1];
    } else {
        const std::vector<std::string> candidates = {
            "data/Dataset_VO",
            "Dataset_VO"
        };

        for (const auto& c : candidates) {
            if (std::filesystem::exists(c) && std::filesystem::is_directory(c)) {
                dataset_path = c;
                break;
            }
        }
    }

    if (dataset_path.empty()) {
        std::cerr
            << "Dataset directory not found.\n"
            << "Expected one of:\n"
            << "  - data/Dataset_VO (recommended)\n"
            << "  - Dataset_VO\n\n"
            << "Run with an explicit path, e.g.:\n"
            << "  ./build/slam_vo_run data/Dataset_VO\n"
            << "  ./build/slam_vo_run /workspace/data/Dataset_VO   (inside Docker)\n";
        return -1;
    }

    std::cout << "Dataset path: " << dataset_path << std::endl;

    // Load image paths
    std::vector<std::string> image_paths = load_image_paths(dataset_path);

    if (image_paths.empty()) {
        std::cerr << "No images found in dataset directory!" << std::endl;
        return -1;
    }

    // Load first image to get dimensions
    cv::Mat first_image = cv::imread(image_paths[0]);
    if (first_image.empty()) {
        std::cerr << "Failed to load first image: " << image_paths[0] << std::endl;
        return -1;
    }

    int image_width = first_image.cols;
    int image_height = first_image.rows;
    std::cout << "Image dimensions: " << image_width << " x " << image_height << std::endl;

    // Initialize Visual Odometry system
    VisualOdometry vo(image_width, image_height);

    // Start trajectory viewer in separate thread
    TrajectoryViewer viewer;
    viewer.init();


    // Create OpenCV windows once
    // cv::namedWindow("Visual Odometry: Frame", cv::WINDOW_NORMAL);
    cv::namedWindow("Visual Odometry: Keypoints", cv::WINDOW_NORMAL);
    // cv::moveWindow("Visual Odometry: Frame", 50, 50);
    cv::moveWindow("Visual Odometry: Keypoints", 50, 500);
    cv::waitKey(1); // optional: forces window creation

    std::cout << "\nProcessing " << image_paths.size() << " frames..." << std::endl;
    std::cout << "Press 'q' or ESC to quit, SPACE to pause/resume" << std::endl;
    std::cout << "========================================" << std::endl;

    bool paused = false;
    int frame_delay = 30;  // ms between frames

    for (size_t i = 0; i < image_paths.size(); ++i) {
        // Check if viewer is still running
        if (viewer.should_quit()) {
            std::cout << "Viewer closed, stopping..." << std::endl;
            break;
        }

        // Load image
        cv::Mat image = cv::imread(image_paths[i]);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_paths[i] << std::endl;
            continue;
        }

        // Create frame
        Frame frame(i, image);

        // Extract timestamp from filename if available
        std::string filename = fs::path(image_paths[i]).stem().string();
        try {
            frame.timestamp = std::stod(filename);
        } catch (...) {
            frame.timestamp = static_cast<double>(i);
        }

        // Process frame
        std::cout << "\n--- Frame " << i + 1 << "/" << image_paths.size() << " ---" << std::endl;
        cv::Mat display_image = vo.process_frame(frame);

        // Update trajectory viewer
        viewer.render_step(vo.get_trajectory());

        // Display images
        // cv::imshow("Visual Odometry: Frame", image);
        cv::imshow("Visual Odometry: Keypoints", display_image);

        // Handle keyboard input
        while (true) {
            int key = cv::waitKey(paused ? 100 : frame_delay);

            if (key == 'q' || key == 'Q' || key == 27) {  // q, Q, or ESC
                std::cout << "\nQuitting..." << std::endl;
                cv::destroyAllWindows();
                return 0;
            }

            if (key == ' ') {  // SPACE
                paused = !paused;
                std::cout << (paused ? "Paused" : "Resumed") << std::endl;
            }

            if (key == '+' || key == '=') {
                frame_delay = std::max(10, frame_delay - 10);
                std::cout << "Frame delay: " << frame_delay << " ms" << std::endl;
            }

            if (key == '-' || key == '_') {
                frame_delay = std::min(500, frame_delay + 10);
                std::cout << "Frame delay: " << frame_delay << " ms" << std::endl;
            }

            if (!paused) {
                break;
            }
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Processing complete!" << std::endl;
    std::cout << "Total frames processed: " << image_paths.size() << std::endl;
    std::cout << "Press any key to exit..." << std::endl;

    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}
