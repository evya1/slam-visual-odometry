
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
    const float x_pan = -2.0f * k;   // try -1*k .. -5*k and tune
    const float eye_x = +2.0f * k; // camera right => scene shifts left
    s_cam_ = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
pangolin::ModelViewLookAt(eye_x, -5*k, -10*k, 0,0,0, pangolin::AxisNegY)
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

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    d_cam_->Activate(s_cam_);

    draw_axes(0.5f);
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
