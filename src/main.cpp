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

#include "frame.h"
#include "trajectory_viewer.h"
#include "visual_odometry.h"

namespace fs = std::filesystem;

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
