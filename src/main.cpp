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
// =============================================================================
// Pose Class - Represents camera pose (rotation + translation)
// =============================================================================
class Pose {
public:
    cv::Mat rotation_matrix;     // 3x3 rotation matrix
    cv::Mat translation_vector;  // 3x1 translation vector

    Pose() {
        rotation_matrix = cv::Mat::eye(3, 3, CV_64F);
        translation_vector = cv::Mat::zeros(3, 1, CV_64F);
    }

    Pose(const cv::Mat& r, const cv::Mat& t) {
        rotation_matrix = r.clone();
        translation_vector = t.clone();
    }

    // Get 4x4 transformation matrix
    cv::Mat get_transformation_matrix() const {
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        rotation_matrix.copyTo(T(cv::Rect(0, 0, 3, 3)));
        translation_vector.copyTo(T(cv::Rect(3, 0, 1, 3)));
        return T;
    }

    // Set from 4x4 transformation matrix
    void set_from_transformation_matrix(const cv::Mat& T) {
        T(cv::Rect(0, 0, 3, 3)).copyTo(rotation_matrix);
        T(cv::Rect(3, 0, 1, 3)).copyTo(translation_vector);
    }

    // Get position (camera center in world coordinates)
    cv::Mat get_position() const {
        return -rotation_matrix.t() * translation_vector;
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
        // Initialize ORB detector with reasonable parameters
        orb_detector = cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

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
        if (frame.image.channels() == 3) {
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
                cv::Mat R_prev = previous_frame.pose.rotation_matrix;
                cv::Mat t_prev = previous_frame.pose.translation_vector;

                // Scale factor (in real system, would come from additional sensors or prior knowledge)
                // For this demo, use a fixed scale
                double scale = 0.1;

                // Update rotation: R_new = R_prev * R
                frame.pose.rotation_matrix = R_prev * R;

                // Update translation: t_new = t_prev + scale * R_prev * t
                frame.pose.translation_vector = t_prev + scale * R_prev * t;

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
private:
    std::thread viewer_thread;
    std::mutex data_mutex;
    std::vector<cv::Mat> trajectory;
    bool should_stop;
    bool running;

public:
    TrajectoryViewer() : should_stop(false), running(false) {}

    ~TrajectoryViewer() {
        stop();
    }

    void start() {
        running = true;
        viewer_thread = std::thread(&TrajectoryViewer::run, this);
    }

    void stop() {
        should_stop = true;
        if (viewer_thread.joinable()) {
            viewer_thread.join();
        }
        running = false;
    }

    void update_trajectory(const std::vector<cv::Mat>& new_trajectory) {
        std::lock_guard<std::mutex> lock(data_mutex);
        trajectory = new_trajectory;
    }

    bool is_running() const {
        return running && !should_stop;
    }

private:
    void run() {
        // Create Pangolin window
        pangolin::CreateWindowAndBind("Visual Odometry: Trajectory", 1024, 768);

        // Enable depth testing
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Define camera render state
        // Looking at origin, camera positioned to see XZ plane (trajectory plane)
        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, pangolin::AxisNegY)
        );

        // Create interactive view
        pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

        while (!pangolin::ShouldQuit() && !should_stop) {
            // Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

            d_cam.Activate(s_cam);

            // Draw coordinate axes at origin
            draw_axes(0.5);

            // Draw grid on XZ plane
            draw_grid();

            // Get current trajectory
            std::vector<cv::Mat> current_trajectory;
            {
                std::lock_guard<std::mutex> lock(data_mutex);
                current_trajectory = trajectory;
            }

            // Draw trajectory
            if (current_trajectory.size() > 1) {
                // Draw trajectory line
                glColor3f(0.0f, 1.0f, 0.0f);  // Green
                glLineWidth(2.0f);
                glBegin(GL_LINE_STRIP);
                for (const auto& pos : current_trajectory) {
                    double x = pos.at<double>(0);
                    double y = pos.at<double>(1);
                    double z = pos.at<double>(2);
                    glVertex3d(x, y, z);
                }
                glEnd();

                // Draw points at each position
                glPointSize(5.0f);
                glBegin(GL_POINTS);
                for (size_t i = 0; i < current_trajectory.size(); ++i) {
                    const auto& pos = current_trajectory[i];
                    double x = pos.at<double>(0);
                    double y = pos.at<double>(1);
                    double z = pos.at<double>(2);

                    if (i == 0) {
                        glColor3f(1.0f, 0.0f, 0.0f);  // Red for start
                    } else if (i == current_trajectory.size() - 1) {
                        glColor3f(0.0f, 0.0f, 1.0f);  // Blue for current
                    } else {
                        glColor3f(0.0f, 1.0f, 0.0f);  // Green for intermediate
                    }
                    glVertex3d(x, y, z);
                }
                glEnd();

                // Draw camera frustum at current position
                if (!current_trajectory.empty()) {
                    const auto& pos = current_trajectory.back();
                    draw_camera_frustum(pos.at<double>(0), pos.at<double>(1), pos.at<double>(2));
                }
            }

            pangolin::FinishFrame();

            // Small delay to prevent excessive CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        pangolin::DestroyWindow("Visual Odometry: Trajectory");
    }

    void draw_axes(float length) {
        glLineWidth(2.0f);
        glBegin(GL_LINES);

        // X axis - Red
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(length, 0, 0);

        // Y axis - Green
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, length, 0);

        // Z axis - Blue
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, length);

        glEnd();
    }

    void draw_grid() {
        glColor3f(0.3f, 0.3f, 0.3f);
        glLineWidth(1.0f);
        glBegin(GL_LINES);

        float grid_size = 10.0f;
        float step = 1.0f;

        for (float i = -grid_size; i <= grid_size; i += step) {
            // Lines parallel to X axis
            glVertex3f(-grid_size, 0, i);
            glVertex3f(grid_size, 0, i);

            // Lines parallel to Z axis
            glVertex3f(i, 0, -grid_size);
            glVertex3f(i, 0, grid_size);
        }

        glEnd();
    }

    void draw_camera_frustum(double x, double y, double z) {
        float size = 0.1f;

        glColor3f(0.0f, 0.5f, 1.0f);  // Light blue
        glLineWidth(2.0f);
        glBegin(GL_LINES);

        // Camera pyramid
        float fx = x, fy = y, fz = z;

        // Front face (towards +Z in camera convention)
        glVertex3f(fx, fy, fz);
        glVertex3f(fx - size, fy - size, fz + size * 2);

        glVertex3f(fx, fy, fz);
        glVertex3f(fx + size, fy - size, fz + size * 2);

        glVertex3f(fx, fy, fz);
        glVertex3f(fx + size, fy + size, fz + size * 2);

        glVertex3f(fx, fy, fz);
        glVertex3f(fx - size, fy + size, fz + size * 2);

        // Base rectangle
        glVertex3f(fx - size, fy - size, fz + size * 2);
        glVertex3f(fx + size, fy - size, fz + size * 2);

        glVertex3f(fx + size, fy - size, fz + size * 2);
        glVertex3f(fx + size, fy + size, fz + size * 2);

        glVertex3f(fx + size, fy + size, fz + size * 2);
        glVertex3f(fx - size, fy + size, fz + size * 2);

        glVertex3f(fx - size, fy + size, fz + size * 2);
        glVertex3f(fx - size, fy - size, fz + size * 2);

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

    // Dataset path
    std::string dataset_path = "NAV_HW2/Dataset_VO";

    if (argc > 1) {
        dataset_path = argv[1];
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
    viewer.start();

    // Create window for keypoints display
    cv::namedWindow("Visual Odometry: Keypoints", cv::WINDOW_AUTOSIZE);

    std::cout << "\nProcessing " << image_paths.size() << " frames..." << std::endl;
    std::cout << "Press 'q' or ESC to quit, SPACE to pause/resume" << std::endl;
    std::cout << "========================================" << std::endl;

    bool paused = false;
    int frame_delay = 30;  // ms between frames

    for (size_t i = 0; i < image_paths.size(); ++i) {
        // Check if viewer is still running
        if (!viewer.is_running()) {
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
        viewer.update_trajectory(vo.get_trajectory());

        // Display keypoints image
        cv::imshow("Visual Odometry: Keypoints", display_image);

        // Handle keyboard input
        while (true) {
            int key = cv::waitKey(paused ? 100 : frame_delay);

            if (key == 'q' || key == 'Q' || key == 27) {  // q, Q, or ESC
                std::cout << "\nQuitting..." << std::endl;
                viewer.stop();
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

    viewer.stop();
    cv::destroyAllWindows();

    return 0;
}
