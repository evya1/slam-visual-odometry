#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <pangolin/pangolin.h>

#include "pose.h"

class TrajectoryViewer {
public:
    TrajectoryViewer() = default;

    void init();

    bool should_quit() const;

    void render_step(const std::vector<Pose> &trajectory);

private:
    bool initialized_ = false;

    pangolin::OpenGlRenderState s_cam_;
    pangolin::View *d_cam_ = nullptr;

    void draw_axes(float axis_display_length);

    void draw_grid();

    void draw_camera_axes_at_pose(const Pose &pose, float axis_len);
};
