#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <stdexcept>

/**
 * @file epipolar_viewer.h
 * @brief Interactive epipolar-geometry viewer for two images.
 *
 * Epipolar geometry (MVG2e §9.2):
 *   Standard form: x2^T F x1 = 0, with l2 = F x1 and l1 = F^T x2.
 *
 * IMPORTANT (this viewer's convention):
 *   This implementation follows the MATLAB vgg_gui_F ordering:
 *     p1^T F p2 = 0
 *   where p1 is in the LEFT image and p2 in the RIGHT image.
 *   Therefore:
 *     - If you click p1 (LEFT), the epipolar line in RIGHT is l2 = F^T p1.
 *     - If you click p2 (RIGHT), the epipolar line in LEFT  is l1 = F p2.
 *
 * 0-based vs 1-based pixels:
 *   OpenCV mouse coordinates are 0-based; MATLAB is typically 1-based.
 *   We convert using x_1based = T x_0based with T = [[1,0,1],[0,1,1],[0,0,1]] and
 *   F transforms as F' = T^{-T} F T^{-1}.
 *
 * @see geometry_conventions.h
 * @see MVG2e Ch. 9 (F/E) and Ch. 11 (normalization, residuals).
 */
class EpipolarViewer {
public:
    EpipolarViewer(cv::Mat left, cv::Mat right, const cv::Matx33d &F_1based,
                   std::string windowName = "Epipolar GUI")
        : imgL_(std::move(left)),
          imgR_(std::move(right)),
          F_(F_1based),
          Ft_(F_1based.t()),
          windowName_(std::move(windowName)) {
        validateInputsOrThrow();

        color_ = cv::Scalar(0, 0, 255); // red (BGR)
        thickness_ = 2;
        activeSide_ = Side::None;

        computeSizes();
        buildCanvasBase();
    }

    // ========= Fundamental-matrix conversion helpers =========

    static cv::Matx33d MakeOneBasedShiftMatrix() {
        return {1, 0, 1,
                           0, 1, 1,
                           0, 0, 1};
    }

    static cv::Matx33d ConvertF_0BasedTo1Based(const cv::Matx33d &F0) {
        const cv::Matx33d T = MakeOneBasedShiftMatrix();
        const cv::Matx33d Tinv = T.inv();
        return Tinv.t() * F0 * Tinv; // T^{-T} * F0 * T^{-1}
    }

    static cv::Matx33d ConvertF_1BasedTo0Based(const cv::Matx33d &F1) {
        const cv::Matx33d T = MakeOneBasedShiftMatrix();
        return T.t() * F1 * T; // T^{T} * F1 * T
    }

    static cv::Matx33d NormalizeFrobenius(const cv::Matx33d &F) {
        double s2 = 0.0;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                s2 += F(r, c) * F(r, c);

        const double n = std::sqrt(s2);
        if (n <= 0.0) return F;
        return 1.0 / n * F;
    }

    static EpipolarViewer CreateWithOpenCV0BasedF(cv::Mat left, cv::Mat right, const cv::Matx33d &F0,
                                                  std::string windowName = "Epipolar GUI",
                                                  bool normalize = false) {
        cv::Matx33d F1 = ConvertF_0BasedTo1Based(F0);
        if (normalize) F1 = NormalizeFrobenius(F1);
        return EpipolarViewer(std::move(left), std::move(right), F1, std::move(windowName));
    }

    void run() {
        cv::namedWindow(windowName_, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(windowName_, &EpipolarViewer::mouseThunk, this);

        showCanvas(canvasBase_);

        while (true) {
            int key = cv::waitKey(20);
            if (key == 27) break; // ESC
            if (key != -1) handleKey(key);
        }
    }

    void setStyle(const cv::Scalar &colorBGR, int thickness) {
        color_ = colorBGR;
        thickness_ = std::max(1, thickness);
    }

private:
    enum class Side { None = -1, Left = 0, Right = 1 };

    struct Segment1Based {
        cv::Point2d a;
        cv::Point2d b;
    };

    cv::Mat imgL_, imgR_;
    cv::Mat canvasBase_, canvasDraw_;

    int wL_ = 0, hL_ = 0;
    int wR_ = 0, hR_ = 0;
    int canvasW_ = 0, canvasH_ = 0;

    cv::Matx33d F_, Ft_;
    Side activeSide_;
    cv::Scalar color_;
    int thickness_;
    std::string windowName_;

    void validateInputsOrThrow() const {
        if (imgL_.empty() || imgR_.empty()) {
            throw std::runtime_error("Images must be non-empty.");
        }
        if (imgL_.type() != imgR_.type()) {
            throw std::runtime_error("Left/right images must have the same cv::Mat type.");
        }
    }

    void computeSizes() {
        hL_ = imgL_.rows;
        wL_ = imgL_.cols;
        hR_ = imgR_.rows;
        wR_ = imgR_.cols;

        canvasW_ = wL_ + wR_;
        canvasH_ = std::max(hL_, hR_);
    }

    void buildCanvasBase() {
        canvasBase_ = cv::Mat::zeros(canvasH_, canvasW_, imgL_.type());
        imgL_.copyTo(canvasBase_(cv::Rect(0, 0, wL_, hL_)));
        imgR_.copyTo(canvasBase_(cv::Rect(wL_, 0, wR_, hR_)));
        canvasDraw_ = canvasBase_.clone();
    }

    void showCanvas(const cv::Mat &canvas) { cv::imshow(windowName_, canvas); }

    bool isInsideLeft(int x, int y) const { return 0 <= x && x < wL_ && 0 <= y && y < hL_; }
    bool isInsideRight(int x, int y) const { return wL_ <= x && x < wL_ + wR_ && 0 <= y && y < hR_; }

    int xShiftFor(Side side) const { return side == Side::Right ? wL_ : 0; }

    cv::Point2d canvasToLocal0Based(Side side, int canvasX, int canvasY) const {
        return {double(canvasX - xShiftFor(side)), double(canvasY)};
    }

    static cv::Vec3d toMatlabHomog1Based(const cv::Point2d &p0) {
        return {p0.x + 1.0, p0.y + 1.0, 1.0};
    }

    cv::Vec3d computeEpipolarLineInOtherImage(Side clickedSide, const cv::Vec3d &p_clicked) const {
        if (clickedSide == Side::Left) return Ft_ * p_clicked;
        if (clickedSide == Side::Right) return F_ * p_clicked;
        return {0, 0, 0};
    }

    static std::optional<Segment1Based> clipLineToImage1Based(const cv::Vec3d &l, int w, int h, double eps = 1e-12) {
        const double a = l[0], b = l[1], c = l[2];
        std::vector<cv::Point2d> pts;

        auto add_if_inside = [&](double x, double y) {
            if (x >= 1.0 && x <= (double) w && y >= 1.0 && y <= (double) h) pts.emplace_back(x, y);
        };

        if (std::abs(b) > eps) {
            add_if_inside(1.0, -(a * 1.0 + c) / b);
            add_if_inside(w, -(a * static_cast<double>(w) + c) / b);
        }
        if (std::abs(a) > eps) {
            add_if_inside(-(b * 1.0 + c) / a, 1.0);
            add_if_inside(-(b * static_cast<double>(h) + c) / a, (double) h);
        }

        std::vector<cv::Point2d> uniq;
        for (const auto &p: pts) {
            bool isNew = true;
            for (const auto &q: uniq) {
                if (std::abs(p.x - q.x) < 1e-7 && std::abs(p.y - q.y) < 1e-7) {
                    isNew = false;
                    break;
                }
            }
            if (isNew) uniq.push_back(p);
        }

        if (uniq.size() < 2) return std::nullopt;
        return Segment1Based{uniq[0], uniq[1]};
    }

    cv::Point2d matlab1BasedToCanvas0Based(const cv::Point2d &p1, Side side) const {
        return {p1.x - 1.0 + xShiftFor(side), p1.y - 1.0};
    }

    void resetDrawLayer() { canvasDraw_ = canvasBase_.clone(); }

    void drawPointMarker(Side side, const cv::Point2d &pLocal0) {
        cv::Point pCanvas((int) std::lround(pLocal0.x + xShiftFor(side)),
                          (int) std::lround(pLocal0.y));
        cv::drawMarker(canvasDraw_, pCanvas, color_, cv::MARKER_CROSS, 14, thickness_);
    }

    void drawEpipolarLineInSide(Side lineSide, const Segment1Based &seg1) {
        cv::Point2d a = matlab1BasedToCanvas0Based(seg1.a, lineSide);
        cv::Point2d b = matlab1BasedToCanvas0Based(seg1.b, lineSide);
        cv::line(canvasDraw_, a, b, color_, thickness_, cv::LINE_AA);
    }

    void updateOverlayFromMouse(Side draggedSide, int canvasX, int canvasY) {
        cv::Point2d pLocal0 = canvasToLocal0Based(draggedSide, canvasX, canvasY);
        cv::Vec3d pMat = toMatlabHomog1Based(pLocal0);

        cv::Vec3d lOther = computeEpipolarLineInOtherImage(draggedSide, pMat);
        Side lineSide = (draggedSide == Side::Left) ? Side::Right : Side::Left;

        int w = lineSide == Side::Left ? wL_ : wR_;
        int h = lineSide == Side::Left ? hL_ : hR_;
        auto seg = clipLineToImage1Based(lOther, w, h);

        resetDrawLayer();
        drawPointMarker(draggedSide, pLocal0);
        if (seg) drawEpipolarLineInSide(lineSide, *seg);

        showCanvas(canvasDraw_);
    }

    void handleMouse(int event, int x, int y, int flags) {
        const bool leftOK = isInsideLeft(x, y);
        const bool rightOK = isInsideRight(x, y);

        if (event == cv::EVENT_LBUTTONDOWN) {
            if (leftOK) {
                activeSide_ = Side::Left;
                updateOverlayFromMouse(activeSide_, x, y);
            } else if (rightOK) {
                activeSide_ = Side::Right;
                updateOverlayFromMouse(activeSide_, x, y);
            }
        }

        if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
            if (activeSide_ == Side::Left && leftOK) updateOverlayFromMouse(activeSide_, x, y);
            else if (activeSide_ == Side::Right && rightOK) {
                updateOverlayFromMouse(activeSide_, x, y);
            }
        }

        if (event == cv::EVENT_LBUTTONUP) activeSide_ = Side::None;
    }

    void handleKey(int key) {
        if (key == 'r') color_ = cv::Scalar(0, 0, 255);
        if (key == 'g') color_ = cv::Scalar(0, 255, 0);
        if (key == 'b') color_ = cv::Scalar(255, 0, 0);
        if (key == 'k') color_ = cv::Scalar(0, 0, 0);

        if (key == '+' || key == '=') thickness_++;
        if (key == '-' || key == '_') thickness_ = std::max(1, thickness_ - 1);
    }

    static void mouseThunk(int event, int x, int y, int flags, void *userdata) {
        auto *self = reinterpret_cast<EpipolarViewer *>(userdata);
        self->handleMouse(event, x, y, flags);
    }
};


enum class FConvention {
    Matlab_1Based,
    OpenCV_0Based
};

inline int run_epipolar_viewer(const cv::Mat &left,
                               const cv::Mat &right,
                               const cv::Matx33d &F_in,
                               FConvention conv = FConvention::OpenCV_0Based,
                               const std::string &windowName = "Epipolar GUI",
                               bool normalizeF = true) {
    try {
        if (left.empty() || right.empty())
            throw std::runtime_error("run_epipolar_viewer: images must be non-empty.");
        if (left.type() != right.type())
            throw std::runtime_error("run_epipolar_viewer: left/right images must have same cv::Mat type.");

        cv::Matx33d F_view = F_in;
        if (conv == FConvention::OpenCV_0Based) {
            F_view = EpipolarViewer::ConvertF_0BasedTo1Based(F_in);
        }

        if (normalizeF) {
            F_view = EpipolarViewer::NormalizeFrobenius(F_view);
        }

        EpipolarViewer viewer(left, right, F_view, windowName);
        viewer.run();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "run_epipolar_viewer error: " << e.what() << "\n";
        return 1;
    }
}