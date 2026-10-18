"""Club head tracking through the golf swing.

Uses background subtraction, edge detection, and motion analysis to track the
club head and detect swing phases.  Works best with a face-on camera angle.
"""

from __future__ import annotations

import math
from typing import Generator

import cv2
import numpy as np

from openlaunch.calibration import Calibration


class ClubTracker:
    """Track the club head through the swing arc."""

    def __init__(self, calibration: Calibration, fps: float) -> None:
        if not calibration.is_calibrated:
            raise RuntimeError("Calibration must be performed before tracking")
        self._cal = calibration
        self._fps = fps  # real-world capture FPS

    # ------------------------------------------------------------------
    # Club head tracking
    # ------------------------------------------------------------------

    def track_club_head(
        self,
        frames: list[np.ndarray] | Generator[np.ndarray, None, None],
        impact_frame: int,
        window: int = 30,
    ) -> list[tuple[float, float, int]]:
        """Track club head through the swing arc (face-on view).

        Strategy:
        1. Background subtraction (MOG2) to isolate moving club.
        2. Edge detection on the moving region.
        3. Find the lowest / most-forward point (club head).
        4. Kalman-filter smoothing.

        Parameters
        ----------
        frames : iterable of BGR frames centred around impact.
        impact_frame : frame index of impact.
        window : number of frames before and after impact to process.

        Returns
        -------
        list of (x_real_inches, y_real_inches, frame_index).
        """
        # Convert generator to list for indexing
        if isinstance(frames, Generator) or not isinstance(frames, list):
            frame_list = list(frames)
        else:
            frame_list = frames

        n = len(frame_list)
        if n == 0:
            return []

        # Background subtractor
        bgsub = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False
        )

        # Kalman filter for smoothing (4-state: x, y, vx, vy)
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kalman.processNoiseCov = 1e-2 * np.eye(4, dtype=np.float32)
        kalman.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)

        positions: list[tuple[float, float, int]] = []
        kalman_initialised = False

        for i, frame in enumerate(frame_list):
            frame_idx = impact_frame - window + i
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Background subtraction
            mask = bgsub.apply(frame)
            # Clean up: morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in foreground mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            club_head_pt: tuple[int, int] | None = None

            if contours:
                # Merge all moving-object contours
                all_pts = np.vstack(contours)
                x_min, y_min, w, h = cv2.boundingRect(all_pts)

                # The club head is typically the lowest (max y) and most
                # forward (max x for right-handed, face-on) point.
                # For a right-handed golfer in face-on view, the club head
                # moves left-to-right through impact.
                # Heuristic: take the rightmost + lowest point.

                # Filter contours with reasonable area
                valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
                if valid_contours:
                    # For each valid contour, find extreme point
                    best_x = -1
                    best_y = -1
                    for c in valid_contours:
                        # Rightmost point
                        rightmost = tuple(c[c[:, :, 0].argmax()][0])
                        # Lowest point
                        bottommost = tuple(c[c[:, :, 1].argmax()][0])
                        # Combine: prefer the point that is both far right and low
                        score = rightmost[0] + bottommost[1] * 0.5
                        if score > best_x + best_y * 0.5:
                            best_x = rightmost[0]
                            best_y = bottommost[1]
                            club_head_pt = (best_x, best_y)

            if club_head_pt is None:
                # Edge-detection fallback
                edges = cv2.Canny(gray, 50, 150)
                edge_contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if edge_contours:
                    largest = max(edge_contours, key=cv2.contourArea)
                    # Bottom-right point of largest edge contour
                    rightmost = tuple(largest[largest[:, :, 0].argmax()][0])
                    club_head_pt = rightmost

            if club_head_pt is not None:
                cx, cy = float(club_head_pt[0]), float(club_head_pt[1])

                # Kalman filter update
                if not kalman_initialised:
                    kalman.statePre = np.array(
                        [[cx], [cy], [0], [0]], dtype=np.float32
                    )
                    kalman.statePost = np.array(
                        [[cx], [cy], [0], [0]], dtype=np.float32
                    )
                    kalman_initialised = True

                measurement = np.array([[cx], [cy]], dtype=np.float32)
                kalman.correct(measurement)
                predicted = kalman.predict()
                smooth_cx = float(predicted[0])
                smooth_cy = float(predicted[1])

                x_real = self._cal.pixels_to_real(smooth_cx)
                y_real = self._cal.pixels_to_real(smooth_cy)
                positions.append((x_real, y_real, frame_idx))

        return positions

    # ------------------------------------------------------------------
    # Speed calculation
    # ------------------------------------------------------------------

    def calculate_club_speed(
        self, positions: list[tuple[float, float, int]], fps: float
    ) -> float:
        """Compute club head speed (peak speed at or just before impact).

        Parameters
        ----------
        positions : list of (x_in, y_in, frame_idx).
        fps : real-world capture FPS.

        Returns
        -------
        float : club head speed in m/s.
        """
        if len(positions) < 3:
            return 0.0

        fps = fps or self._fps

        # Compute per-frame velocities
        speeds: list[float] = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            dist_in = math.hypot(dx, dy)
            dt = (positions[i][2] - positions[i - 1][2]) / fps
            if dt < 1e-9:
                continue
            speed_in_per_s = dist_in / dt
            speed_m_per_s = speed_in_per_s * 0.0254
            speeds.append(speed_m_per_s)

        if not speeds:
            return 0.0

        # Peak speed is typically at or just before impact.
        # Use the 90th-percentile speed to reduce noise influence.
        speeds.sort()
        idx90 = max(0, int(len(speeds) * 0.9) - 1)
        return speeds[idx90]

    # ------------------------------------------------------------------
    # Swing phase detection
    # ------------------------------------------------------------------

    def detect_swing_phases(
        self, positions: list[tuple[float, float, int]], fps: float
    ) -> dict[str, int]:
        """Detect key swing phases from club head trajectory.

        Returns
        -------
        dict with keys:
            'address' : frame (minimal motion)
            'backswing_top' : frame (direction reversal upward)
            'downswing_start' : frame (acceleration begins)
            'impact' : frame (max speed)
            'follow_through' : frame (deceleration / direction change)
        """
        if len(positions) < 5:
            return {
                "address": 0,
                "backswing_top": 0,
                "downswing_start": 0,
                "impact": 0,
                "follow_through": 0,
            }

        fps = fps or self._fps

        # Compute velocities (x and y separately)
        vx: list[float] = []
        vy: list[float] = []
        speed_list: list[float] = []
        frame_indices: list[int] = []

        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            dt = (positions[i][2] - positions[i - 1][2]) / fps
            if dt < 1e-9:
                continue
            vx.append(dx / dt)
            vy.append(dy / dt)
            speed_list.append(math.hypot(dx, dy) / dt)
            frame_indices.append(positions[i][2])

        if len(speed_list) < 3:
            return {
                "address": positions[0][2],
                "backswing_top": positions[0][2],
                "downswing_start": positions[0][2],
                "impact": positions[0][2],
                "follow_through": positions[-1][2],
            }

        # Smooth speed signal
        speed_arr = np.array(speed_list, dtype=np.float64)
        k = max(3, len(speed_arr) // 20)
        if k % 2 == 0:
            k += 1
        speed_smooth = np.convolve(speed_arr, np.ones(k) / k, mode="same")

        # 1. Impact = frame of maximum speed
        impact_i = int(np.argmax(speed_smooth))
        impact_frame = frame_indices[impact_i]

        # 2. Address = earliest frame with minimal motion (before backswing)
        # Find first frame where speed exceeds a threshold
        threshold = float(np.max(speed_smooth)) * 0.05
        motion_start = 0
        for j, s in enumerate(speed_smooth):
            if s > threshold:
                motion_start = j
                break
        address_frame = frame_indices[max(0, motion_start)]

        # 3. Backswing top = direction reversal (y-velocity changes sign)
        # In face-on view, backswing moves club upward (decreasing y in image),
        # transition = vy goes from negative to positive.
        vy_arr = np.array(vy, dtype=np.float64)
        # Search before impact for vy sign change
        for j in range(impact_i - 1, 0, -1):
            if vy_arr[j] < 0 and (j + 1 < len(vy_arr)) and vy_arr[j + 1] >= 0:
                backswing_top_frame = frame_indices[j]
                break
        else:
            # Fallback: mid-point between address and impact
            mid = max(0, motion_start + (impact_i - motion_start) // 2)
            backswing_top_frame = frame_indices[mid]

        # 4. Downswing start = shortly after backswing top (acceleration begins)
        # Find where speed starts increasing after the top
        for j in range(impact_i):
            if speed_smooth[j] > threshold and speed_smooth[j + 1] > speed_smooth[j]:
                downswing_start_frame = frame_indices[j]
                break
        else:
            downswing_start_frame = backswing_top_frame

        # 5. Follow through = speed drops below 50% of peak after impact
        half_peak = float(np.max(speed_smooth)) * 0.5
        follow_through_frame = frame_indices[-1]
        for j in range(impact_i, len(speed_smooth)):
            if speed_smooth[j] < half_peak:
                follow_through_frame = frame_indices[j]
                break

        return {
            "address": address_frame,
            "backswing_top": backswing_top_frame,
            "downswing_start": downswing_start_frame,
            "impact": impact_frame,
            "follow_through": follow_through_frame,
        }


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    from openlaunch.calibration import Calibration

    cal = Calibration()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cal.calibrate_from_ball(frame, ball_center=(500, 540), ball_edge=(600, 540))

    tracker = ClubTracker(cal, fps=240.0)

    # Simulate positions: backswing then downswing
    n = 60
    sim_positions = []
    for i in range(n):
        # Backswing: x decreases, y decreases (up in image)
        if i < 25:
            x = 50.0 - i * 0.8
            y = 30.0 - i * 0.4
        elif i < 30:
            # Transition
            x = 50.0 - 25 * 0.8 + (i - 25) * 0.3
            y = 30.0 - 25 * 0.4 + (i - 25) * 0.2
        else:
            # Downswing: accelerating
            t = i - 30
            x = 50.0 - 25 * 0.8 + 5 * 0.3 + t * (1.0 + t * 0.15)
            y = 30.0 - 25 * 0.4 + 5 * 0.2 + t * 0.5
        sim_positions.append((x, y, i))

    speed = tracker.calculate_club_speed(sim_positions, fps=240)
    phases = tracker.detect_swing_phases(sim_positions, fps=240)
    print(f"Club speed: {speed:.2f} m/s ({speed * 2.23694:.1f} mph)")
    print(f"Phases: {phases}")
