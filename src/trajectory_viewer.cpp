
#include "trajectory_viewer.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <algorithm>
#include <filesystem>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
    cv::Vec3d mat3x1_to_vec3d(const cv::Mat &m) {
        return {m.at<double>(0), m.at<double>(1), m.at<double>(2)};
    }

    double norm3(const cv::Vec3d &v) {
        return std::sqrt(v.dot(v));
    }

    cv::Vec3d normalize_or_fallback(const cv::Vec3d &v, const cv::Vec3d &fallback) {
        const double n = norm3(v);
        if (n < 1e-12) return fallback;
        return (1.0 / n) * v;
    }

    bool save_bound_window_jpeg(const std::string &filepath) {
        GLint viewport[4] = {0, 0, 0, 0};
        glGetIntegerv(GL_VIEWPORT, viewport);
        const int w = viewport[2];
        const int h = viewport[3];

        if (w <= 0 || h <= 0) {
            std::cerr << "[TrajectoryViewer] Invalid GL viewport for screenshot: "
                    << w << "x" << h << "\n";
            return false;
        }

        cv::Mat rgb(h, w, CV_8UC3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        // After pangolin::FinishFrame(), the rendered image is in the front buffer.
        glReadBuffer(GL_FRONT);
        glFinish();
        glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);

        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        cv::flip(bgr, bgr, 0); // OpenGL origin is bottom-left

        const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 95};
        const bool ok = cv::imwrite(filepath, bgr, params);
        if (!ok) {
            std::cerr << "[TrajectoryViewer] Failed to write screenshot: " << filepath << "\n";
            return false;
        }
        return true;
    }
}

void TrajectoryViewer::init() {
    if (initialized_) return;

    pangolin::CreateWindowAndBind("Visual Odometry: Trajectory", 1024, 768);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const float k = 0.1f;

    const float eye_x = +2.0f * k; // camera right => scene shifts left
    s_cam_ = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(eye_x, -5 * k, -10 * k, 0, 0, 0, pangolin::AxisNegY)
    );

    d_cam_ = &pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam_));

    initialized_ = true;
}

bool TrajectoryViewer::should_quit() {
    return pangolin::ShouldQuit();
}

void TrajectoryViewer::render_step(const std::vector<Pose> &trajectory) {
    init();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    d_cam_->Activate(s_cam_);

    // World reference frame
    draw_axes(0.5f);
    draw_grid();

    if (trajectory.empty()) {
        pangolin::FinishFrame();
        return;
    }

    // Trajectory polyline (needs at least 2 poses)
    if (trajectory.size() > 1) {
        glColor3f(0.0f, 1.0f, 0.0f);
        glLineWidth(2.0f);

        glBegin(GL_LINE_STRIP);
        for (const auto &p: trajectory) {
            const cv::Mat& pos = p.C_w();
            glVertex3d(pos.at<double>(0), pos.at<double>(1), pos.at<double>(2));
        }
        glEnd();
    }

    // Trajectory points
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const cv::Mat pos = trajectory[i].C_w();
        if (i == 0) glColor3f(1.0f, 0.0f, 0.0f); // start: red
        else if (i == trajectory.size() - 1) glColor3f(0.0f, 0.0f, 1.0f); // end: blue
        else glColor3f(0.0f, 1.0f, 0.0f); // middle: green
        glVertex3d(pos.at<double>(0), pos.at<double>(1), pos.at<double>(2));
    }
    glEnd();

    // Draw current camera pose (axes + frustum)
    {
        const Pose &cur = trajectory.back();
        draw_camera_axes_at_pose(cur, 0.3f);
        draw_camera_frustum_at_pose(cur, 0.25f);
    }

    // Draw history every N poses (axes + frustum)
    {
        const int kEveryN = 10;
        for (size_t i = 0; i < trajectory.size(); i += kEveryN) {
            draw_camera_axes_at_pose(trajectory[i], 0.1f);
            draw_camera_frustum_at_pose(trajectory[i], 0.08f);
        }
    }

    pangolin::FinishFrame();
}

bool TrajectoryViewer::save_trajectory_screenshots(const std::vector<Pose> &trajectory,
                                                   const std::string &out_dir) {
    init();

    if (TrajectoryViewer::should_quit()) {
        std::cerr << "[TrajectoryViewer] Pangolin window already closed; cannot screenshot.\n";
        return false;
    }
    if (trajectory.empty()) {
        std::cerr << "[TrajectoryViewer] Empty trajectory; nothing to screenshot.\n";
        return false;
    }

    std::filesystem::create_directories(out_dir);

    cv::Vec3d mn(+1e30, +1e30, +1e30);
    cv::Vec3d mx(-1e30, -1e30, -1e30);
    for (const auto &p: trajectory) {
        const cv::Vec3d v = mat3x1_to_vec3d(p.C_w());
        mn[0] = std::min(mn[0], v[0]);
        mn[1] = std::min(mn[1], v[1]);
        mn[2] = std::min(mn[2], v[2]);
        mx[0] = std::max(mx[0], v[0]);
        mx[1] = std::max(mx[1], v[1]);
        mx[2] = std::max(mx[2], v[2]);
    }

    const cv::Vec3d center = 0.5 * (mn + mx);
    const double extent = std::max({mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]});
    const double dist = std::max(2.5 * extent, 1.0);

    const pangolin::OpenGlMatrix orig_mv = s_cam_.GetModelViewMatrix();

    struct ViewSpec {
        const char *tag;
        cv::Vec3d dir;
        pangolin::AxisDirection up;
    };

    const std::vector<ViewSpec> views = {
        {"posX", cv::Vec3d(+1, 0, 0), pangolin::AxisNegY},
        {"negX", cv::Vec3d(-1, 0, 0), pangolin::AxisNegY},
        {"posY", cv::Vec3d(0, +1, 0), pangolin::AxisZ},
        {"negY", cv::Vec3d(0, -1, 0), pangolin::AxisZ},
        {"posZ", cv::Vec3d(0, 0, +1), pangolin::AxisNegY},
        {"negZ", cv::Vec3d(0, 0, -1), pangolin::AxisNegY},
        {"iso", cv::Vec3d(+1, -1, -1), pangolin::AxisNegY},
    };

    bool all_ok = true;

    for (const auto &v: views) {
        const cv::Vec3d dir = normalize_or_fallback(v.dir, cv::Vec3d(0, 0, -1));
        const cv::Vec3d eye = center + dist * dir;

        s_cam_.SetModelViewMatrix(
            pangolin::ModelViewLookAt(
                eye[0], eye[1], eye[2],
                center[0], center[1], center[2],
                v.up
            )
        );

        // Render with this camera pose
        render_step(trajectory);

        const std::filesystem::path out_path =
                std::filesystem::path(out_dir) /
                (std::string("trajectory_view_from_") + v.tag + ".jpg");

        all_ok = all_ok && save_bound_window_jpeg(out_path.string());
    }

    // Restore interactive view
    s_cam_.SetModelViewMatrix(orig_mv);
    render_step(trajectory);

    return all_ok;
}

void TrajectoryViewer::draw_axes(float axis_display_length) {
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(axis_display_length, 0, 0);
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, axis_display_length, 0);
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, axis_display_length);
    glEnd();
}

void TrajectoryViewer::draw_grid() {
    glColor3f(0.3f, 0.3f, 0.3f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    const float grid_size = 10.0f;
    const float step = 1.0f;
    for (float i = -grid_size; i <= grid_size; i += step) {
        glVertex3f(-grid_size, 0, i);
        glVertex3f(grid_size, 0, i);
        glVertex3f(i, 0, -grid_size);
        glVertex3f(i, 0, grid_size);
    }
    glEnd();
}

void TrajectoryViewer::draw_camera_axes_at_pose(const Pose &pose, float axis_len) {
    const cv::Mat& C = pose.C_w();
    const double cx = C.at<double>(0);
    const double cy = C.at<double>(1);
    const double cz = C.at<double>(2);

    // Camera axes in world coords are the columns of R_wc.
    // For visualization, we draw "forward" as -Z_c (OpenGL-style), hence the minus.
    const cv::Vec3d xw(pose.R_wc.at<double>(0, 0), pose.R_wc.at<double>(1, 0), pose.R_wc.at<double>(2, 0));
    const cv::Vec3d yw(pose.R_wc.at<double>(0, 1), pose.R_wc.at<double>(1, 1), pose.R_wc.at<double>(2, 1));
    const cv::Vec3d zw = -cv::Vec3d(
        pose.R_wc.at<double>(0, 2),
        pose.R_wc.at<double>(1, 2),
        pose.R_wc.at<double>(2, 2)
    );

    glLineWidth(3.0f);
    glBegin(GL_LINES);

    // X axis (red)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3d(cx, cy, cz);
    glVertex3d(cx + axis_len * xw[0], cy + axis_len * xw[1], cz + axis_len * xw[2]);

    // Y axis (green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3d(cx, cy, cz);
    glVertex3d(cx + axis_len * yw[0], cy + axis_len * yw[1], cz + axis_len * yw[2]);

    // Z axis (blue)  [visual forward is -Zc, already built into zw]
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3d(cx, cy, cz);
    glVertex3d(cx + axis_len * zw[0], cy + axis_len * zw[1], cz + axis_len * zw[2]);

    glEnd();
}

static cv::Vec3d cam_to_world(const Pose &pose, const cv::Vec3d &Xc) {
    return {
        pose.R_wc.at<double>(0, 0) * Xc[0] + pose.R_wc.at<double>(0, 1) * Xc[1] + pose.R_wc.at<double>(0, 2) * Xc[2] +
        pose.t_wc.at<double>(0),
        pose.R_wc.at<double>(1, 0) * Xc[0] + pose.R_wc.at<double>(1, 1) * Xc[1] + pose.R_wc.at<double>(1, 2) * Xc[2] +
        pose.t_wc.at<double>(1),
        pose.R_wc.at<double>(2, 0) * Xc[0] + pose.R_wc.at<double>(2, 1) * Xc[1] + pose.R_wc.at<double>(2, 2) * Xc[2] +
        pose.t_wc.at<double>(2)
    };
}

void TrajectoryViewer::draw_camera_frustum_at_pose(const Pose &pose, float scale) {
    // Camera coordinate model (in camera coords):
    // origin at (0,0,0), looking along -Zc (your convention for visualization)
    const double d = 1.0 * scale; // depth
    const double hw = 0.6 * scale; // half width
    const double hh = 0.4 * scale; // half height

    const cv::Vec3d O(0, 0, 0);
    const cv::Vec3d C1(-hw, -hh, -d);
    const cv::Vec3d C2(+hw, -hh, -d);
    const cv::Vec3d C3(+hw, +hh, -d);
    const cv::Vec3d C4(-hw, +hh, -d);

    const auto Ow = cam_to_world(pose, O);
    const auto P1w = cam_to_world(pose, C1);
    const auto P2w = cam_to_world(pose, C2);
    const auto P3w = cam_to_world(pose, C3);
    const auto P4w = cam_to_world(pose, C4);

    glLineWidth(1.5f);
    glColor3f(1.0f, 1.0f, 0.0f); // yellow frustum

    glBegin(GL_LINES);

    // Rays from camera center to image-plane corners
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(P1w[0], P1w[1], P1w[2]);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(P2w[0], P2w[1], P2w[2]);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(P3w[0], P3w[1], P3w[2]);
    glVertex3d(Ow[0], Ow[1], Ow[2]);
    glVertex3d(P4w[0], P4w[1], P4w[2]);

    // Rectangle (image plane)
    glVertex3d(P1w[0], P1w[1], P1w[2]);
    glVertex3d(P2w[0], P2w[1], P2w[2]);
    glVertex3d(P2w[0], P2w[1], P2w[2]);
    glVertex3d(P3w[0], P3w[1], P3w[2]);
    glVertex3d(P3w[0], P3w[1], P3w[2]);
    glVertex3d(P4w[0], P4w[1], P4w[2]);
    glVertex3d(P4w[0], P4w[1], P4w[2]);
    glVertex3d(P1w[0], P1w[1], P1w[2]);

    glEnd();
}
