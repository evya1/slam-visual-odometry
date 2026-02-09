#pragma once

#include <vector>
#include <string>

#include <pangolin/pangolin.h>

#include "pose.h"

class TrajectoryViewer {
public:
    TrajectoryViewer() = default;

    void init();

    static bool should_quit();

    void render_step(const std::vector<Pose> &trajectory);
    static void draw_camera_frustum_at_pose(const Pose &pose, float scale);
    bool save_trajectory_screenshots(const std::vector<Pose> &trajectory,
                                     const std::string &out_dir = "trajectory_screenshots");

private:
    bool initialized_ = false;

    pangolin::OpenGlRenderState s_cam_;
    pangolin::View *d_cam_ = nullptr;

    static void draw_axes(float axis_display_length);

    static void draw_grid();

    static void draw_camera_axes_at_pose(const Pose &pose, float axis_len);
};
