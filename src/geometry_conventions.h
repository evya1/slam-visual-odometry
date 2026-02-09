#pragma once
/**
 * @file geometry_conventions.h
 * @brief Project-wide conventions (aligned with MVG2e Ch. 9–12).
 *
 * ---- Image points (homogeneous) ----
 * Pixel point in OpenCV coordinates:
 *   x = [u, v, 1]^T , with u in [0..W-1], v in [0..H-1].
 *
 *
 * ---- Fundamental matrix (MVG2e §9.2) ----
 * Standard MVG form:
 *   x2^T F x1 = 0  (x1 in image1, x2 in image2)
 * and epipolar lines:
 *   l2 = F x1,   l1 = F^T x2.
 *
 * ---- Essential matrix (MVG2e §9.6) ----
 * With intrinsics K and normalized image coordinates x̂ = K^{-1} x:
 *   x̂2^T E x̂1 = 0,   E = [t]_x R.
 *
 * ---- Pose stored in this project ----
 * Pose is camera->world (T_wc):
 *   x_w = R_wc x_c + t_wc,   camera center C_w = t_wc.
 *
 * MVG camera matrix typically uses world->camera:
 *   P = K [R_cw | t_cw],  where R_cw = R_wc^T and t_cw = -R_wc^T t_wc.
 */
