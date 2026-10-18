"""Ball detection and tracking for golf launch metrics.

Detects the golf ball using HoughCircles + white-colour filtering, then tracks
it across post-impact frames to compute launch speed, angle, and direction.
"""

from __future__ import annotations

import math
from typing import Generator

import cv2
import numpy as np

from openlaunch.calibration import Calibration, GOLF_BALL_DIAMETER_INCHES


class BallTracker:
    """Track a golf ball across video frames to measure launch conditions."""

    def __init__(self, calibration: Calibration, fps: float) -> None:
        if not calibration.is_calibrated:
            raise RuntimeError("Calibration must be performed before tracking")
        self._cal = calibration
        self._fps = fps  # real-world FPS (capture rate)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_ball(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int, float] | None:
        """Detect a golf ball in a single frame.

        Strategy:
        1. Grayscale + blur.
        2. HoughCircles with radius range derived from calibration.
        3. Filter candidates by white colour.
        4. Return most likely ball (cx, cy, radius).

        Parameters
        ----------
        frame : BGR image.
        roi : optional (x, y, w, h) region of interest.

        Returns
        -------
        (cx, cy, radius) in pixel coordinates, or None.
        """
        working = frame.copy()
        if roi is not None:
            x, y, w, h = roi
            working = working[y : y + h, x : x + w]
        else:
            x = y = 0

        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Expected ball radius in pixels from calibration
        expected_r = self._cal.real_to_pixels(GOLF_BALL_DIAMETER_INCHES) / 2.0
        min_r = max(4, int(expected_r * 0.5))
        max_r = max(min_r + 4, int(expected_r * 2.0))

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(30, int(expected_r)),
            param1=80,
            param2=25,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is None or len(circles) == 0:
            # Fallback: relax parameters
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.5,
                minDist=20,
                param1=60,
                param2=20,
                minRadius=max(3, min_r - 2),
                maxRadius=max_r + 10,
            )
            if circles is None or len(circles) == 0:
                return None

        # Filter by whiteness (golf balls are white)
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        candidates = []
        for cx, cy, r in circles[0]:
            cx_i, cy_i, r_i = int(cx), int(cy), int(r)
            # Sample a small patch around the circle centre
            h_img, w_img = hsv.shape[:2]
            patch_r = max(2, int(r_i * 0.6))
            y1 = max(0, cy_i - patch_r)
            y2 = min(h_img, cy_i + patch_r)
            x1 = max(0, cx_i - patch_r)
            x2 = min(w_img, cx_i + patch_r)
            patch = hsv[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # White: high value, low saturation
            mean_v = float(np.mean(patch[:, :, 2]))
            mean_s = float(np.mean(patch[:, :, 1]))
            whiteness = mean_v / 255.0 * (1.0 - mean_s / 255.0)

            # Scoring: closeness to expected radius + whiteness
            radius_score = 1.0 - abs(r_i - expected_r) / (expected_r + 1.0)
            score = 0.5 * radius_score + 0.5 * whiteness
            candidates.append((cx + x, cy + y, r, score))

        if not candidates:
            return None

        # Return the highest-scoring candidate
        best = max(candidates, key=lambda c: c[3])
        return (int(best[0]), int(best[1]), float(best[2]))

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def track_ball(
        self,
        frames: Generator[np.ndarray, None, None] | list[np.ndarray],
        start_frame: int,
        n_frames: int = 5,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[float, float, int]]:
        """Track the ball across *n_frames* after impact.

        Parameters
        ----------
        frames : iterable of BGR frames (starting at impact or just after).
        start_frame : frame index of the first frame in *frames*.
        n_frames : how many frames to track.
        roi : optional region of interest for detection.

        Returns
        -------
        list of (x_real_inches, y_real_inches, frame_index).
        """
        positions: list[tuple[float, float, int]] = []
        prev_center: tuple[int, int] | None = None

        for i, frame in enumerate(frames):
            if i >= n_frames:
                break

            det = self.detect_ball(frame, roi=roi)
            if det is not None:
                cx, cy, _r = det
                x_real = self._cal.pixels_to_real(cx)
                y_real = self._cal.pixels_to_real(cy)
                positions.append((x_real, y_real, start_frame + i))
                prev_center = (cx, cy)
            elif prev_center is not None:
                # Optical-flow fallback: search a small window around prev pos
                x0 = max(0, prev_center[0] - 40)
                y0 = max(0, prev_center[1] - 40)
                small_roi = (x0, y0, 80, 80)
                det2 = self.detect_ball(frame, roi=small_roi)
                if det2 is not None:
                    cx, cy, _r = det2
                    x_real = self._cal.pixels_to_real(cx)
                    y_real = self._cal.pixels_to_real(cy)
                    positions.append((x_real, y_real, start_frame + i))
                    prev_center = (cx, cy)
                # else: ball lost, skip frame

        return positions

    # ------------------------------------------------------------------
    # Metric calculations
    # ------------------------------------------------------------------

    def calculate_ball_speed(self, positions: list[tuple[float, float, int]], fps: float) -> float:
        """Compute initial ball speed from the first few tracked positions.

        Uses the first 2–3 frames immediately after impact (before drag
        significantly decelerates the ball).

        Parameters
        ----------
        positions : list of (x_in, y_in, frame_idx).
        fps : real-world frames per second.

        Returns
        -------
        float : ball speed in m/s.
        """
        if len(positions) < 2:
            return 0.0

        fps = fps or self._fps

        # Use up to the first 3 frames for initial speed
        n = min(3, len(positions))
        total_dist_in = 0.0
        for i in range(1, n):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            total_dist_in += math.hypot(dx, dy)

        # Time span in seconds
        frame_span = positions[n - 1][2] - positions[0][2]
        if frame_span <= 0:
            return 0.0

        dt = frame_span / fps
        if dt < 1e-9:
            return 0.0

        speed_in_per_s = total_dist_in / dt
        speed_m_per_s = speed_in_per_s * 0.0254  # inches → meters
        return speed_m_per_s

    def calculate_launch_angle(
        self, positions: list[tuple[float, float, int]], fps: float
    ) -> float:
        """Compute launch angle from face-on view.

        Angle of initial ball flight relative to horizontal.  In the image
        coordinate system, +y is downward; we flip for real-world convention.

        Returns
        -------
        float : launch angle in degrees (positive = upward).
        """
        if len(positions) < 2:
            return 0.0

        # Use first 2 frames for direction
        dx = positions[1][0] - positions[0][0]  # horizontal (inches)
        dy = positions[0][1] - positions[1][1]  # vertical (flip y: up = positive)

        angle = math.degrees(math.atan2(dy, abs(dx) + 1e-9))
        return angle

    def calculate_launch_direction(
        self, positions: list[tuple[float, float, int]], fps: float
    ) -> float:
        """Compute launch direction from down-the-line view.

        Positive = right (push / slice), negative = left (pull / draw).

        Returns
        -------
        float : direction in degrees relative to target line.
        """
        if len(positions) < 2:
            return 0.0

        # Down-the-line: lateral movement is the *y* pixel axis in image
        # because the camera looks along the target line.
        # In a DTL view, horizontal pixel axis ≈ target line,
        # vertical pixel axis ≈ away/toward camera.
        # Lateral deviation shows up as a shift in the *x* (left-right) axis
        # of the image — which maps to our *y_real* in DTL orientation.

        # Use first 2 frames
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]

        # Lateral deviation = dy in down-the-line view
        # Forward motion = dx
        direction = math.degrees(math.atan2(dy, abs(dx) + 1e-9))
        return direction


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ms_to_mph(speed_ms: float) -> float:
    """Convert m/s to mph."""
    return speed_ms * 2.23694


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    from openlaunch.calibration import Calibration

    # Synthetic: 200 px = 1.68" ball diameter → ppi ≈ 119.05
    cal = Calibration()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cal.calibrate_from_ball(frame, ball_center=(500, 540), ball_edge=(600, 540))
    print(f"Calibration: {cal}")

    tracker = BallTracker(cal, fps=240.0)
    # Simulated positions: ball moving right and slightly up
    pos = [
        (0.0, 0.0, 0),
        (1.0, 0.2, 1),
        (2.0, 0.4, 2),
    ]
    speed = tracker.calculate_ball_speed(pos, fps=240)
    angle = tracker.calculate_launch_angle(pos, fps=240)
    print(f"Simulated speed: {speed:.2f} m/s ({_ms_to_mph(speed):.1f} mph)")
    print(f"Simulated launch angle: {angle:.1f}°")
