
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

    // Pan target to the LEFT -> scene appears more to the RIGHT
    // const float x_pan = -2.0f * k;   // try -1*k .. -5*k and tune
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

    draw_axes(0.5f);
    draw_grid();

    if (trajectory.size() > 1) {
        glColor3f(0.0f, 1.0f, 0.0f);
        glLineWidth(2.0f);

        glBegin(GL_LINE_STRIP);
        for (const auto &p: trajectory) {
            const cv::Mat pos = p.get_position();
            glVertex3d(pos.at<double>(0), pos.at<double>(1), pos.at<double>(2));
        }
        glEnd();

        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < trajectory.size(); ++i) {
            const cv::Mat pos = trajectory[i].get_position();
            if (i == 0) glColor3f(1.0f, 0.0f, 0.0f);
            else if (i == trajectory.size() - 1) glColor3f(0.0f, 0.0f, 1.0f);
            else glColor3f(0.0f, 1.0f, 0.0f);
            glVertex3d(pos.at<double>(0), pos.at<double>(1), pos.at<double>(2));
        }
        glEnd();

        // Draw orientation for the LAST pose (current camera)
        draw_camera_axes_at_pose(trajectory.back(), 0.3f);

        // Optional: draw smaller orientation every N poses to see turning history
        const int kEveryN = 10;
        for (size_t i = 0; i < trajectory.size(); i += kEveryN) {
            draw_camera_axes_at_pose(trajectory[i], 0.1f);
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

    // Columns of R_wc are the camera axes expressed in world coordinates.
    const cv::Vec3d xw(pose.R.at<double>(0, 0), pose.R.at<double>(1, 0), pose.R.at<double>(2, 0));
    const cv::Vec3d yw(pose.R.at<double>(0, 1), pose.R.at<double>(1, 1), pose.R.at<double>(2, 1));
    const cv::Vec3d zw = -cv::Vec3d(
        pose.R.at<double>(0, 2),
        pose.R.at<double>(1, 2),
        pose.R.at<double>(2, 2)
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

    // Z axis (blue) = forward arrow
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3d(cx, cy, cz);
    glVertex3d(cx + axis_len * zw[0], cy + axis_len * zw[1], cz + axis_len * zw[2]);

    glEnd();
}

