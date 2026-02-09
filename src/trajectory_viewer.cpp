
#include "trajectory_viewer.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

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

#if 0
    {
        // DEBUG: identity pose (R_wc = I) shifted so it doesn't overlap the world axes.
        Pose p;
        p.t_vec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
        draw_camera_axes_at_pose(p, 0.5f);
        draw_camera_frustum_at_pose(p, 0.4f);
    }
#endif

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
            const cv::Mat pos = p.get_position();
            glVertex3d(pos.at<double>(0), pos.at<double>(1), pos.at<double>(2));
        }
        glEnd();
    }

    // Trajectory points
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const cv::Mat pos = trajectory[i].get_position();
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
    const cv::Mat C = pose.get_position();
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

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3d(cx, cy, cz);
    glVertex3d(cx + axis_len * zw[0], cy + axis_len * zw[1], cz + axis_len * zw[2]);

    glEnd();
}


static cv::Vec3d cam_to_world(const Pose &pose, const cv::Vec3d &Xc) {
    // Xw = R_wc * Xc + t_wc
    return cv::Vec3d(
        pose.R_wc.at<double>(0, 0) * Xc[0] + pose.R_wc.at<double>(0, 1) * Xc[1] + pose.R_wc.at<double>(0, 2) * Xc[2] +
        pose.t_vec.at<double>(0),
        pose.R_wc.at<double>(1, 0) * Xc[0] + pose.R_wc.at<double>(1, 1) * Xc[1] + pose.R_wc.at<double>(1, 2) * Xc[2] +
        pose.t_vec.at<double>(1),
        pose.R_wc.at<double>(2, 0) * Xc[0] + pose.R_wc.at<double>(2, 1) * Xc[1] + pose.R_wc.at<double>(2, 2) * Xc[2] +
        pose.t_vec.at<double>(2)
    );
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

    const cv::Vec3d Ow = cam_to_world(pose, O);
    const cv::Vec3d P1w = cam_to_world(pose, C1);
    const cv::Vec3d P2w = cam_to_world(pose, C2);
    const cv::Vec3d P3w = cam_to_world(pose, C3);
    const cv::Vec3d P4w = cam_to_world(pose, C4);

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

