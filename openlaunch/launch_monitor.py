"""Main launch monitor: orchestrates video, calibration, tracking, and pose
analysis to produce a complete :class:`LaunchResult`.

Typical usage::

    from openlaunch.launch_monitor import LaunchMonitor

    lm = LaunchMonitor("swing_240fps.mov", club_length_inches=45.75, camera_angle="face_on")
    result = lm.analyze()
    print(result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from openlaunch.video import VideoReader
from openlaunch.calibration import Calibration
from openlaunch.ball_tracker import BallTracker
from openlaunch.club_tracker import ClubTracker
from openlaunch.pose_estimator import SwingPoseEstimator, SwingAnalysis


# ------------------------------------------------------------------
# Conversion constants
# ------------------------------------------------------------------

_MS_TO_MPH = 2.23694
_M_TO_YDS = 1.09361
_IN_TO_M = 0.0254


# ------------------------------------------------------------------
# Launch result
# ------------------------------------------------------------------

@dataclass
class LaunchResult:
    """Complete launch monitor output."""

    ball_speed_ms: float = 0.0
    ball_speed_mph: float = 0.0
    club_speed_ms: float = 0.0
    club_speed_mph: float = 0.0
    smash_factor: float = 0.0
    launch_angle_deg: float = 0.0
    launch_direction_deg: float = 0.0
    carry_distance_yds: float = 0.0
    total_distance_yds: float = 0.0
    spin_rate_rpm: float = 0.0
    descent_angle_deg: float = 0.0
    apex_height_m: float = 0.0
    swing_analysis: SwingAnalysis = field(default_factory=SwingAnalysis)
    trajectory_points: list[tuple[float, float, float]] = field(default_factory=list)
    # (x_m, y_m, t_s) trajectory from physics model

    # Diagnostic / debugging
    impact_frame: int = -1
    ball_positions: list[tuple[float, float, int]] = field(default_factory=list)
    club_positions: list[tuple[float, float, int]] = field(default_factory=list)
    calibration_ppi: float = 0.0

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "═══════════════════════════════════════",
            "       OPENLAUNCH  LAUNCH  MONITOR      ",
            "═══════════════════════════════════════",
            "",
            f"  Ball Speed      : {self.ball_speed_mph:6.1f} mph  ({self.ball_speed_ms:.1f} m/s)",
            f"  Club Speed      : {self.club_speed_mph:6.1f} mph  ({self.club_speed_ms:.1f} m/s)",
            f"  Smash Factor    : {self.smash_factor:6.2f}",
            "",
            f"  Launch Angle    : {self.launch_angle_deg:6.1f}°",
            f"  Launch Direction: {self.launch_direction_deg:6.1f}°",
            f"  Spin Rate       : {self.spin_rate_rpm:6.0f} rpm",
            "",
            f"  Carry Distance  : {self.carry_distance_yds:6.1f} yds",
            f"  Total Distance  : {self.total_distance_yds:6.1f} yds",
            f"  Apex Height     : {self.apex_height_m:6.1f} m",
            f"  Descent Angle   : {self.descent_angle_deg:6.1f}°",
            "",
            "─── Swing Analysis ───────────────────",
            f"  Shoulder Tilt (addr) : {self.swing_analysis.shoulder_tilt_at_address:.1f}°",
            f"  Shoulder Tilt (top)  : {self.swing_analysis.shoulder_tilt_at_top:.1f}°",
            f"  Hip Rotation (impact): {self.swing_analysis.hip_rotation_at_impact:.1f}°",
            f"  Weight Shift Ratio   : {self.swing_analysis.weight_shift_ratio:.2f}",
            f"  Spine Angle         : {self.swing_analysis.spine_angle:.1f}°",
            f"  Arm Extension       : {self.swing_analysis.arm_extension_at_impact:.0%}",
            f"  Head Stability      : {self.swing_analysis.head_stability:.1f}\"",
            f"  Swing Tempo         : {self.swing_analysis.swing_tempo:.1f}:1",
        ]

        tips = self.swing_analysis.get_coaching_tips()
        if tips:
            lines.append("")
            lines.append("─── Coaching Tips ─────────────────────")
            for tip in tips:
                lines.append(f"  • {tip}")

        lines.append("═══════════════════════════════════════")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "ball_speed_ms": self.ball_speed_ms,
            "ball_speed_mph": self.ball_speed_mph,
            "club_speed_ms": self.club_speed_ms,
            "club_speed_mph": self.club_speed_mph,
            "smash_factor": self.smash_factor,
            "launch_angle_deg": self.launch_angle_deg,
            "launch_direction_deg": self.launch_direction_deg,
            "carry_distance_yds": self.carry_distance_yds,
            "total_distance_yds": self.total_distance_yds,
            "spin_rate_rpm": self.spin_rate_rpm,
            "descent_angle_deg": self.descent_angle_deg,
            "apex_height_m": self.apex_height_m,
            "swing_analysis": {
                "shoulder_tilt_at_address": self.swing_analysis.shoulder_tilt_at_address,
                "shoulder_tilt_at_top": self.swing_analysis.shoulder_tilt_at_top,
                "hip_rotation_at_impact": self.swing_analysis.hip_rotation_at_impact,
                "weight_shift_ratio": self.swing_analysis.weight_shift_ratio,
                "spine_angle": self.swing_analysis.spine_angle,
                "arm_extension_at_impact": self.swing_analysis.arm_extension_at_impact,
                "head_stability": self.swing_analysis.head_stability,
                "swing_tempo": self.swing_analysis.swing_tempo,
                "coaching_tips": self.swing_analysis.get_coaching_tips(),
            },
            "impact_frame": self.impact_frame,
            "calibration_ppi": self.calibration_ppi,
        }


# ------------------------------------------------------------------
# Launch monitor
# ------------------------------------------------------------------

class LaunchMonitor:
    """Orchestrate the full video → launch-metrics pipeline."""

    def __init__(
        self,
        video_path: str,
        club_length_inches: float = 45.75,
        camera_angle: str = "face_on",
        real_fps: float | None = None,
    ) -> None:
        self._video_path = video_path
        self._club_length = club_length_inches
        self._camera_angle = camera_angle
        self._real_fps = real_fps

    def analyze(
        self,
        impact_frame: int | None = None,
        calibration_points: dict[str, Any] | None = None,
        ball_roi: tuple[int, int, int, int] | None = None,
    ) -> LaunchResult:
        """Run the full analysis pipeline.

        Steps:
        1. Load video and read metadata.
        2. Calibrate (from provided points, or auto-detect).
        3. Find impact frame (if not provided).
        4. Track ball after impact.
        5. Track club through swing.
        6. Estimate pose for swing analysis.
        7. Calculate launch metrics.
        8. Estimate carry / trajectory from physics model.
        9. Return complete LaunchResult.

        Parameters
        ----------
        impact_frame : explicit impact frame index, or None to auto-detect.
        calibration_points : dict with keys 'top_point' / 'sole_point' for
            club calibration, or 'ball_center' / 'ball_edge' for ball
            calibration.  If None, attempts auto-calibration.
        ball_roi : optional (x, y, w, h) for ball detection.

        Returns
        -------
        LaunchResult
        """
        result = LaunchResult()

        # ── 1. Load video ────────────────────────────────────────────
        try:
            video = VideoReader(self._video_path, real_fps=self._real_fps)
        except (FileNotFoundError, IOError) as exc:
            result._error = str(exc)  # type: ignore[attr-defined]
            return result

        fps = video.real_fps if video.real_fps > 0 else video.fps

        # ── 2. Calibrate ─────────────────────────────────────────────
        cal = Calibration(
            club_length_inches=self._club_length,
            camera_angle=self._camera_angle,
        )

        calibrated = False
        if calibration_points is not None:
            ref_frame = video.get_frame(0)
            if "top_point" in calibration_points and "sole_point" in calibration_points:
                try:
                    cal.calibrate_from_frame(
                        ref_frame,
                        top_point=calibration_points["top_point"],
                        sole_point=calibration_points["sole_point"],
                    )
                    calibrated = True
                except (ValueError, RuntimeError):
                    pass
            elif "ball_center" in calibration_points and "ball_edge" in calibration_points:
                try:
                    cal.calibrate_from_ball(
                        ref_frame,
                        ball_center=calibration_points["ball_center"],
                        ball_edge=calibration_points["ball_edge"],
                    )
                    calibrated = True
                except (ValueError, RuntimeError):
                    pass

        if not calibrated:
            # Auto-calibrate: try to find a ball in the first frame
            ref_frame = video.get_frame(0)
            try:
                ppi = cal.auto_calibrate_from_ball(ref_frame, roi=ball_roi)
                if ppi is not None and ppi > 0:
                    calibrated = True
            except Exception:
                pass

        if not calibrated:
            # Last resort: assume a reasonable PPI based on resolution
            # Typical: a driver fills ~30% of a 1080p frame vertically
            # 45.75 inches ≈ 0.3 * 1080 = 324 px → ppi ≈ 7.08
            cal.pixels_per_inch = max(1.0, video.resolution[1] * 0.3 / self._club_length)
            calibrated = True

        result.calibration_ppi = cal.pixels_per_inch

        # ── 3. Find impact frame ─────────────────────────────────────
        if impact_frame is None:
            impact_frame = video.find_impact_frame(roi=ball_roi)
        result.impact_frame = impact_frame

        # ── 4. Track ball after impact ───────────────────────────────
        ball_positions: list[tuple[float, float, int]] = []
        ball_speed_ms = 0.0
        launch_angle = 0.0
        launch_direction = 0.0

        if cal.is_calibrated and impact_frame < video.frame_count:
            try:
                bt = BallTracker(cal, fps)
                start = impact_frame
                end = min(impact_frame + 10, video.frame_count)
                frames_gen = video.get_frames_range(start, end)
                ball_positions = bt.track_ball(
                    frames_gen, start_frame=start, n_frames=10, roi=ball_roi
                )
                result.ball_positions = ball_positions

                if len(ball_positions) >= 2:
                    ball_speed_ms = bt.calculate_ball_speed(ball_positions, fps)
                    launch_angle = bt.calculate_launch_angle(ball_positions, fps)
                    launch_direction = bt.calculate_launch_direction(ball_positions, fps)
            except Exception:
                # Graceful degradation: continue without ball data
                pass

        result.ball_speed_ms = ball_speed_ms
        result.ball_speed_mph = ball_speed_ms * _MS_TO_MPH
        result.launch_angle_deg = launch_angle
        result.launch_direction_deg = launch_direction

        # ── 5. Track club through swing ──────────────────────────────
        club_speed_ms = 0.0
        club_positions: list[tuple[float, float, int]] = []
        swing_phases: dict[str, int] = {}

        if cal.is_calibrated:
            try:
                ct = ClubTracker(cal, fps)
                window = 30
                club_start = max(0, impact_frame - window)
                club_end = min(impact_frame + window, video.frame_count)
                club_frames = list(video.get_frames_range(club_start, club_end))
                club_positions = ct.track_club_head(
                    club_frames, impact_frame=impact_frame, window=window
                )
                result.club_positions = club_positions

                if len(club_positions) >= 3:
                    club_speed_ms = ct.calculate_club_speed(club_positions, fps)
                    swing_phases = ct.detect_swing_phases(club_positions, fps)
            except Exception:
                pass

        result.club_speed_ms = club_speed_ms
        result.club_speed_mph = club_speed_ms * _MS_TO_MPH

        # ── 6. Pose / swing analysis ────────────────────────────────
        swing_analysis = SwingAnalysis()
        try:
            pose_estimator = SwingPoseEstimator()
            pose_start = max(0, impact_frame - 60)
            pose_end = min(impact_frame + 30, video.frame_count)
            pose_frames = list(video.get_frames_range(pose_start, pose_end))
            relative_impact = impact_frame - pose_start
            swing_analysis = pose_estimator.analyze_swing(pose_frames, relative_impact)
            pose_estimator.close()
        except (ImportError, Exception):
            # MediaPipe not available or failed — graceful degradation
            pass

        result.swing_analysis = swing_analysis

        # ── 7. Derived metrics ───────────────────────────────────────

        # Smash factor = ball_speed / club_speed
        if club_speed_ms > 0:
            result.smash_factor = ball_speed_ms / club_speed_ms
        elif ball_speed_ms > 0:
            # Estimate club speed from typical smash factor (1.45 for driver)
            estimated_smash = 1.45 if self._club_length > 40 else 1.30
            result.club_speed_ms = ball_speed_ms / estimated_smash
            result.club_speed_mph = result.club_speed_ms * _MS_TO_MPH
            result.smash_factor = estimated_smash

        # Spin rate estimate (empirical formula)
        # Spin ≈ f(club_speed, launch_angle, club type)
        # Driver: ~1500-3500 rpm, Irons: ~4000-8000 rpm
        result.spin_rate_rpm = self._estimate_spin_rate(
            ball_speed_ms, club_speed_ms, launch_angle
        )

        # ── 8. Trajectory estimation ─────────────────────────────────
        try:
            # Try the physics module for accurate carry first
            from openlaunch.physics import BallFlightSimulator as _BFS, LaunchConditions as _LC

            bfs = _BFS()
            lc = _LC(
                velocity=ball_speed_ms,
                launch_angle=math.radians(launch_angle),
                spin_rate=result.spin_rate_rpm,
            )
            tr = bfs.simulate_trajectory(lc, max_time=12.0, dt=0.05)
            result.carry_distance_yds = tr.carry_distance * _M_TO_YDS
            result.total_distance_yds = tr.total_distance * _M_TO_YDS
            result.apex_height_m = tr.max_height
            result.descent_angle_deg = tr.descent_angle
            # Convert trajectory points for export
            result.trajectory_points = [
                (float(pt.position[0]), float(pt.position[2]), pt.t)
                for pt in tr.points
            ]
        except (ImportError, Exception):
            # Fallback: simplified in-house model
            try:
                trajectory = self._compute_trajectory(
                    ball_speed_ms=ball_speed_ms,
                    launch_angle_deg=launch_angle,
                    spin_rate_rpm=result.spin_rate_rpm,
                    club_length_inches=self._club_length,
                )
                result.trajectory_points = trajectory

                if trajectory:
                    carry_m = 0.0
                    total_m = 0.0
                    apex_m = 0.0
                    for x, y, _t in trajectory:
                        if y > apex_m:
                            apex_m = y
                        if y >= 0:
                            carry_m = x
                    total_m = trajectory[-1][0] if trajectory else carry_m

                    result.carry_distance_yds = carry_m * _M_TO_YDS
                    result.total_distance_yds = total_m * _M_TO_YDS
                    result.apex_height_m = apex_m

                    if len(trajectory) >= 2:
                        for i in range(1, len(trajectory)):
                            if trajectory[i][1] < 0:
                                dx = trajectory[i][0] - trajectory[i - 1][0]
                                dy = trajectory[i][1] - trajectory[i - 1][1]
                                result.descent_angle_deg = abs(
                                    math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9))
                                )
                                break
            except Exception:
                pass

        # ── Done ─────────────────────────────────────────────────────
        return result

    # ------------------------------------------------------------------
    # Spin rate estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_spin_rate(
        ball_speed_ms: float,
        club_speed_ms: float,
        launch_angle_deg: float,
    ) -> float:
        """Estimate spin rate from launch conditions.

        Empirical approximation based on:
        - Higher club speed → higher spin (for irons)
        - Higher launch angle (with same speed) → more loft → more spin
        - Drivers produce less spin than irons

        Returns RPM.
        """
        if ball_speed_ms <= 0 and club_speed_ms <= 0:
            return 0.0

        # Use whichever speed is available
        speed = max(ball_speed_ms, club_speed_ms)
        if speed <= 0:
            return 0.0

        # Base spin rate (rough empirical model)
        # Driver: ~25 rpm per mph of ball speed at typical launch
        # Iron:  ~45 rpm per mph
        speed_mph = speed * _MS_TO_MPH

        # Heuristic: lower launch angle = driver = less spin
        if launch_angle_deg < 15:
            # Driver-like
            base_rpm_per_mph = 25.0
        elif launch_angle_deg < 25:
            # Fairway wood / hybrid
            base_rpm_per_mph = 35.0
        else:
            # Iron
            base_rpm_per_mph = 50.0

        spin = speed_mph * base_rpm_per_mph

        # Adjust: higher launch → more loft → more spin
        spin *= (1.0 + launch_angle_deg / 50.0)

        return max(0.0, spin)

    # ------------------------------------------------------------------
    # Trajectory computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trajectory(
        ball_speed_ms: float,
        launch_angle_deg: float,
        spin_rate_rpm: float,
        club_length_inches: float = 45.75,
        dt: float = 0.02,
        max_time: float = 12.0,
    ) -> list[tuple[float, float, float]]:
        """Compute ball trajectory using a simplified physics model.

        Uses Magnus effect (spin) and aerodynamic drag.

        Parameters
        ----------
        ball_speed_ms : initial ball speed in m/s.
        launch_angle_deg : launch angle in degrees.
        spin_rate_rpm : backspin in RPM.
        club_length_inches : used to infer club type for drag model.
        dt : time step in seconds.
        max_time : maximum simulation time.

        Returns
        -------
        list of (x_m, y_m, t_s) — horizontal distance, height, time.
        """
        if ball_speed_ms <= 0:
            return []

        try:
            from openlaunch.physics import BallFlightSimulator, LaunchConditions

            sim = BallFlightSimulator()
            conditions = LaunchConditions(
                velocity=ball_speed_ms,
                launch_angle=math.radians(launch_angle_deg),  # physics API expects radians
                spin_rate=spin_rate_rpm,
            )
            tr = sim.simulate_trajectory(conditions, max_time=max_time, dt=dt)
            # Convert TrajectoryResult → list of (x_m, height_m, t_s)
            trajectory = []
            for pt in tr.points:
                x_m = float(pt.position[0])
                z_m = float(pt.position[2])  # z = height
                trajectory.append((x_m, z_m, pt.t))
            return trajectory
        except (ImportError, Exception):
            pass

        # Fallback: simplified in-house trajectory model
        angle_rad = math.radians(launch_angle_deg)
        vx = ball_speed_ms * math.cos(angle_rad)
        vy = ball_speed_ms * math.sin(angle_rad)

        # Golf ball parameters
        mass = 0.04593  # kg
        radius = 0.02135  # m (1.68" diameter / 2)
        area = math.pi * radius ** 2
        rho = 1.225  # air density kg/m³
        cd = 0.25  # drag coefficient (golf ball with dimples)
        cl = 0.15  # lift coefficient (simplified Magnus)

        g = 9.81  # m/s²

        # Spin in rad/s
        omega = spin_rate_rpm * 2.0 * math.pi / 60.0

        x = 0.0
        y = 0.0
        t = 0.0

        trajectory: list[tuple[float, float, float]] = [(x, y, t)]

        while t < max_time and y >= -0.01:
            speed = math.hypot(vx, vy)
            if speed < 0.1:
                break

            # Drag force (opposing velocity)
            drag = 0.5 * rho * cd * area * speed ** 2
            ax_drag = -drag * vx / (speed * mass)
            ay_drag = -drag * vy / (speed * mass)

            # Magnus (lift) force — perpendicular to velocity, upward for backspin
            magnus = 0.5 * rho * cl * area * speed * (omega * radius)
            # Direction: perpendicular to velocity, in the plane of motion
            # For backspin, lift is upward (positive y) when ball moves forward
            if speed > 0:
                perp_x = -vy / speed
                perp_y = vx / speed
            else:
                perp_x = 0.0
                perp_y = 1.0

            ax_magnus = magnus * perp_x / mass
            ay_magnus = magnus * perp_y / mass

            # Total acceleration
            ax = ax_drag + ax_magnus
            ay = ay_drag + ay_magnus - g

            # Euler integration
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            t += dt

            trajectory.append((x, y, t))

            if y < -0.01 and t > 0.5:
                break

        return trajectory


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Demo with no video file: show what LaunchResult looks like
        demo = LaunchResult(
            ball_speed_ms=62.0,
            ball_speed_mph=62.0 * _MS_TO_MPH,
            club_speed_ms=42.7,
            club_speed_mph=42.7 * _MS_TO_MPH,
            smash_factor=62.0 / 42.7,
            launch_angle_deg=11.5,
            launch_direction_deg=-1.2,
            carry_distance_yds=245.0,
            total_distance_yds=268.0,
            spin_rate_rpm=2650,
            descent_angle_deg=38.0,
            apex_height_m=28.0,
        )
        print(demo.summary())
        print("\n--- Dict export ---")
        import json
        print(json.dumps(demo.to_dict(), indent=2))
        sys.exit(0)

    # With a video file
    lm = LaunchMonitor(sys.argv[1], club_length_inches=45.75, camera_angle="face_on")
    result = lm.analyze()
    print(result.summary())
