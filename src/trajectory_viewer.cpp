
#include "trajectory_viewer.h"

#include <cmath>   // std::sin, std::cos, M_PI (on some systems)
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

    s_cam_ = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(
            0, -1.5, -3.0,   // eye
            0,  0.0,  0.0,   // target
            pangolin::AxisNegY
        )
    );

    d_cam_ = &pangolin::CreateDisplay()
                 .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                 .SetHandler(new pangolin::Handler3D(s_cam_));

    initialized_ = true;
}

bool TrajectoryViewer::should_quit() const {
    return pangolin::ShouldQuit();
}

void TrajectoryViewer::render_step(const std::vector<cv::Mat>& trajectory) {
    init();

    // Center the initial view on the current trajectory (run once)
    if (!home_set_ && !trajectory.empty()) {
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

        s_cam_.SetModelViewMatrix(
            pangolin::ModelViewLookAt(
                cx + dx_rot, cy + dy, cz + dz_rot,  // eye
                cx,          cy,      cz,           // target
                pangolin::AxisNegY
            )
        );
        home_set_ = true;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    d_cam_->Activate(s_cam_);

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

    pangolin::FinishFrame();
}

void TrajectoryViewer::draw_axes(float axis_display_length) {
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(0, 0, 0); glVertex3f(axis_display_length, 0, 0);
    glColor3f(0.0f, 1.0f, 0.0f); glVertex3f(0, 0, 0); glVertex3f(0, axis_display_length, 0);
    glColor3f(0.0f, 0.0f, 1.0f); glVertex3f(0, 0, 0); glVertex3f(0, 0, axis_display_length);
    glEnd();
}

void TrajectoryViewer::draw_grid() {
    glColor3f(0.3f, 0.3f, 0.3f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    const float grid_size = 10.0f;
    const float step = 1.0f;
    for (float i = -grid_size; i <= grid_size; i += step) {
        glVertex3f(-grid_size, 0, i); glVertex3f(grid_size, 0, i);
        glVertex3f(i, 0, -grid_size); glVertex3f(i, 0, grid_size);
    }
    glEnd();
}
