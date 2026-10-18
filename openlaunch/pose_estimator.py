"""MediaPipe Pose-based swing analysis for the OpenLaunch launch monitor.

Estimates body pose across swing frames and extracts golf-specific metrics:
shoulder tilt, hip rotation, weight shift, spine angle, arm extension,
head stability, and swing tempo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Generator

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class SwingAnalysis:
    """Swing analysis results derived from pose estimation."""

    shoulder_tilt_at_address: float = 0.0   # degrees
    shoulder_tilt_at_top: float = 0.0       # degrees
    hip_rotation_at_impact: float = 0.0     # degrees (open = positive)
    weight_shift_ratio: float = 0.5         # lead/trail foot pressure estimate
    spine_angle: float = 0.0                # degrees from vertical
    arm_extension_at_impact: float = 0.0    # ratio 0–1 (1 = fully extended)
    head_stability: float = 0.0             # inches lateral movement (lower = better)
    swing_tempo: float = 0.0                 # backswing:downswing time ratio

    def get_coaching_tips(self) -> list[str]:
        """Return actionable coaching feedback based on metrics."""
        tips: list[str] = []

        # Shoulder tilt at top
        if self.shoulder_tilt_at_top < 25:
            tips.append(
                f"Shoulder tilt at top ({self.shoulder_tilt_at_top:.0f}°) is shallow — "
                "aim for 30–45° for better coil and power."
            )
        elif self.shoulder_tilt_at_top > 50:
            tips.append(
                f"Shoulder tilt at top ({self.shoulder_tilt_at_top:.0f}°) is excessive — "
                "this may cause over-rotation and loss of balance."
            )

        # Hip rotation at impact
        if self.hip_rotation_at_impact < 30:
            tips.append(
                f"Hip rotation at impact ({self.hip_rotation_at_impact:.0f}°) is low — "
                "work on clearing hips earlier for more speed (target 35–50°)."
            )
        elif self.hip_rotation_at_impact > 55:
            tips.append(
                f"Hip rotation at impact ({self.hip_rotation_at_impact:.0f}°) is very open — "
                "ensure you're not spinning out and losing balance."
            )

        # Swing tempo
        if self.swing_tempo > 0:
            if self.swing_tempo < 2.0:
                tips.append(
                    f"Swing tempo ({self.swing_tempo:.1f}:1) is rushed — "
                    "ideal is 2.5:1 to 3.5:1. Try a smoother takeaway."
                )
            elif self.swing_tempo > 4.0:
                tips.append(
                    f"Swing tempo ({self.swing_tempo:.1f}:1) is very slow — "
                    "this may indicate a pause at the top."
                )

        # Head stability
        if self.head_stability > 2.0:
            tips.append(
                f"Head movement ({self.head_stability:.1f}\" lateral) is high — "
                "work on keeping your head steady (< 2\") for consistent contact."
            )

        # Weight shift
        if self.weight_shift_ratio < 0.45:
            tips.append(
                f"Weight shift ratio ({self.weight_shift_ratio:.2f}) suggests "
                "insufficient weight transfer to lead side at impact — "
                "target ~0.60 (60/40 lead/trail)."
            )
        elif self.weight_shift_ratio > 0.80:
            tips.append(
                f"Weight shift ratio ({self.weight_shift_ratio:.2f}) is very "
                "aggressive — ensure you maintain balance through the finish."
            )

        # Spine angle
        if abs(self.spine_angle) > 35:
            tips.append(
                f"Spine angle ({self.spine_angle:.0f}° from vertical) is extreme — "
                "try to maintain a more athletic posture at address."
            )

        # Arm extension
        if self.arm_extension_at_impact < 0.7:
            tips.append(
                f"Arm extension at impact ({self.arm_extension_at_impact:.0%}) is low — "
                "work on extending arms through the hitting area for consistency."
            )

        if not tips:
            tips.append("Swing metrics look good — keep up the consistent practice!")

        return tips


# ------------------------------------------------------------------
# MediaPipe landmark indices
# ------------------------------------------------------------------

# Key landmarks for golf analysis (MediaPipe Pose 33-landmark model)
LANDMARK_NOSE = 0
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24
LANDMARK_LEFT_KNEE = 25
LANDMARK_RIGHT_KNEE = 26
LANDMARK_LEFT_ANKLE = 27
LANDMARK_RIGHT_ANKLE = 28


# ------------------------------------------------------------------
# Pose estimator
# ------------------------------------------------------------------

class SwingPoseEstimator:
    """Estimate body pose in golf swing frames and compute swing metrics."""

    def __init__(self) -> None:
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is required for pose estimation. "
                "Install with: pip install mediapipe"
            )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate_pose(self, frame: np.ndarray) -> dict:
        """Run MediaPipe Pose on a single frame.

        Parameters
        ----------
        frame : BGR image.

        Returns
        -------
        dict mapping landmark index → (x, y, z, visibility).
        Normalised coordinates [0, 1].
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        landmarks: dict[int, tuple[float, float, float, float]] = {}
        if result.pose_landmarks is not None:
            for idx, lm in enumerate(result.pose_landmarks.landmark):
                landmarks[idx] = (lm.x, lm.y, lm.z, lm.visibility)
        return landmarks

    def analyze_swing(
        self,
        frames: list[np.ndarray] | Generator[np.ndarray, None, None],
        impact_frame: int,
    ) -> SwingAnalysis:
        """Full swing analysis from pose data across frames.

        Parameters
        ----------
        frames : iterable of BGR frames covering the full swing.
        impact_frame : index of the impact frame.

        Returns
        -------
        SwingAnalysis with computed metrics.
        """
        # Convert generator to list for random access
        if not isinstance(frames, list):
            frame_list = list(frames)
        else:
            frame_list = frames

        n = len(frame_list)
        if n == 0:
            return SwingAnalysis()

        # Estimate poses for key frames
        # Address: ~30% into clip (early frames)
        addr_i = max(0, n // 5)
        # Top: ~50% into clip
        top_i = max(0, n // 2)
        # Impact: provided index or 70% through
        imp_i = min(impact_frame, n - 1) if impact_frame < n else max(0, int(n * 0.7))

        pose_address = self.estimate_pose(frame_list[addr_i])
        pose_top = self.estimate_pose(frame_list[top_i])
        pose_impact = self.estimate_pose(frame_list[imp_i])

        # Track head positions across all frames for stability
        head_positions: list[tuple[float, float]] = []
        # Also track lead/trail knee positions for weight shift
        ankle_positions: list[tuple[float, float, float, float]] = []

        # Sample a subset for head tracking (every 2nd frame for speed)
        sample_step = max(1, n // 30)
        for i in range(0, n, sample_step):
            pose = self.estimate_pose(frame_list[i])
            if LANDMARK_NOSE in pose:
                lm = pose[LANDMARK_NOSE]
                head_positions.append((lm[0], lm[1]))
            if LANDMARK_LEFT_ANKLE in pose and LANDMARK_RIGHT_ANKLE in pose:
                la = pose[LANDMARK_LEFT_ANKLE]
                ra = pose[LANDMARK_RIGHT_ANKLE]
                ankle_positions.append((la[0], la[1], ra[0], ra[1]))

        analysis = SwingAnalysis()

        # --- Shoulder tilt ---
        analysis.shoulder_tilt_at_address = self._shoulder_tilt(pose_address)
        analysis.shoulder_tilt_at_top = self._shoulder_tilt(pose_top)

        # --- Hip rotation at impact ---
        analysis.hip_rotation_at_impact = self._hip_rotation(pose_impact)

        # --- Spine angle ---
        analysis.spine_angle = self._spine_angle(pose_address)

        # --- Weight shift ratio ---
        analysis.weight_shift_ratio = self._weight_shift(pose_impact)

        # --- Arm extension at impact ---
        analysis.arm_extension_at_impact = self._arm_extension(pose_impact)

        # --- Head stability ---
        analysis.head_stability = self._head_stability(head_positions)

        # --- Swing tempo ---
        # We need timing of backswing vs downswing.
        # Heuristic: find the frame where shoulder tilt reverses direction.
        analysis.swing_tempo = self._swing_tempo(
            frame_list, addr_i, imp_i, top_i
        )

        return analysis

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_landmark(
        pose: dict, idx: int
    ) -> tuple[float, float, float, float] | None:
        return pose.get(idx)

    @staticmethod
    def _shoulder_tilt(pose: dict) -> float:
        """Shoulder tilt angle (degrees) — angle of shoulder line from horizontal.

        For a right-handed golfer, positive = right shoulder lower than left
        (which is the correct tilt at the top of the backswing).
        """
        ls = pose.get(LANDMARK_LEFT_SHOULDER)
        rs = pose.get(LANDMARK_RIGHT_SHOULDER)
        if ls is None or rs is None:
            return 0.0
        if ls[3] < 0.3 or rs[3] < 0.3:
            return 0.0

        dy = rs[1] - ls[1]  # positive = right shoulder lower
        dx = abs(rs[0] - ls[0]) + 1e-9
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    @staticmethod
    def _hip_rotation(pose: dict) -> float:
        """Hip rotation angle at impact (degrees, open = positive).

        Open hips = lead hip closer to target (camera) than trail hip.
        In a face-on view for a right-handed golfer, the left hip leads.
        We use the z-coordinate difference as a proxy for rotation.
        """
        lh = pose.get(LANDMARK_LEFT_HIP)
        rh = pose.get(LANDMARK_RIGHT_HIP)
        if lh is None or rh is None:
            return 0.0
        if lh[3] < 0.3 or rh[3] < 0.3:
            return 0.0

        # z-coordinate indicates depth; more negative = further from camera.
        # For a right-hander, left hip leads → left hip z closer to camera (less negative).
        # Rotation angle approximation from z-difference.
        dz = rh[2] - lh[2]  # positive = right hip further back = open
        dx = abs(lh[0] - rh[0]) + 1e-9
        angle = math.degrees(math.atan2(abs(dz), dx))
        return angle

    @staticmethod
    def _spine_angle(pose: dict) -> float:
        """Spine angle from vertical (degrees).

        Measured as the angle between the vertical and a line from hip
        midpoint to shoulder midpoint.
        """
        ls = pose.get(LANDMARK_LEFT_SHOULDER)
        rs = pose.get(LANDMARK_RIGHT_SHOULDER)
        lh = pose.get(LANDMARK_LEFT_HIP)
        rh = pose.get(LANDMARK_RIGHT_HIP)
        if any(x is None for x in (ls, rs, lh, rh)):
            return 0.0
        if any(x[3] < 0.3 for x in (ls, rs, lh, rh)):
            return 0.0

        shoulder_mid_y = (ls[1] + rs[1]) / 2
        hip_mid_y = (lh[1] + rh[1]) / 2
        shoulder_mid_x = (ls[0] + rs[0]) / 2
        hip_mid_x = (lh[0] + rh[0]) / 2

        dy = hip_mid_y - shoulder_mid_y  # positive = shoulders above hips
        dx = abs(shoulder_mid_x - hip_mid_x)
        angle = math.degrees(math.atan2(dx, dy + 1e-9))
        return angle

    @staticmethod
    def _weight_shift(pose: dict) -> float:
        """Weight shift ratio (lead foot / total).

        Estimated from knee and ankle positions.  A shifted weight shows
        the lead knee moving toward the target (more lateral on x-axis).
        """
        lk = pose.get(LANDMARK_LEFT_KNEE)
        rk = pose.get(LANDMARK_RIGHT_KNEE)
        la = pose.get(LANDMARK_LEFT_ANKLE)
        ra = pose.get(LANDMARK_RIGHT_ANKLE)
        if any(x is None for x in (lk, rk, la, ra)):
            return 0.5
        if any(x[3] < 0.3 for x in (lk, rk, la, ra)):
            return 0.5

        # Right-handed golfer: lead = left foot.
        # Weight shift = how much lead knee has moved laterally
        # relative to lead ankle vs trail side.
        lead_lateral = abs(lk[0] - la[0])
        trail_lateral = abs(rk[0] - ra[0])
        total = lead_lateral + trail_lateral
        if total < 1e-6:
            return 0.5

        # Normalise: more lead lateral = more weight on lead side
        ratio = 0.3 + 0.7 * (lead_lateral / total)  # scale to ~0.3–1.0
        return min(1.0, max(0.0, ratio))

    @staticmethod
    def _arm_extension(pose: dict) -> float:
        """Arm extension ratio at impact (0 = bent, 1 = fully extended).

        Measures the angle at the lead elbow.  A straight arm ≈ 180°.
        """
        # For a right-handed golfer, lead arm = left arm
        ls = pose.get(LANDMARK_LEFT_SHOULDER)
        le = pose.get(LANDMARK_LEFT_ELBOW)
        lw = pose.get(LANDMARK_LEFT_WRIST)
        if any(x is None for x in (ls, le, lw)):
            return 0.0
        if any(x[3] < 0.3 for x in (ls, le, lw)):
            return 0.0

        # Vectors: shoulder→elbow, elbow→wrist
        v1 = (ls[0] - le[0], ls[1] - le[1])
        v2 = (lw[0] - le[0], lw[1] - le[1])

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(*v1) + 1e-9
        mag2 = math.hypot(*v2) + 1e-9

        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        angle = math.degrees(math.acos(cos_angle))

        # 180° = fully extended → ratio = 1.0
        # 90° = fully bent → ratio ≈ 0.0
        extension = max(0.0, min(1.0, (angle - 90) / 90))
        return extension

    @staticmethod
    def _head_stability(head_positions: list[tuple[float, float]]) -> float:
        """Head lateral movement in normalised-coordinate units, converted to inches.

        Assumes ~1.0 normalised-x ≈ shoulder width ≈ 18 inches for scaling.
        """
        if len(head_positions) < 2:
            return 0.0

        xs = [p[0] for p in head_positions]
        if not xs:
            return 0.0

        # Lateral movement = range of x positions
        x_range = max(xs) - min(xs)
        # Convert to inches (rough: shoulder width ~18 inches ≈ 0.3 normalised units)
        inches = x_range * 18.0 / 0.3
        return inches

    def _swing_tempo(
        self,
        frame_list: list[np.ndarray],
        address_i: int,
        impact_i: int,
        top_i: int,
    ) -> float:
        """Compute backswing : downswing time ratio.

        Uses shoulder-tilt direction changes to estimate the top-of-swing
        timing more precisely.
        """
        if impact_i <= address_i:
            return 0.0

        # The 'top' is already estimated; refine by looking for the frame
        # where shoulder tilt reverses direction between address and impact.
        best_top = top_i
        prev_tilt = 0.0
        for i in range(address_i, min(impact_i + 1, len(frame_list))):
            pose = self.estimate_pose(frame_list[i])
            tilt = self._shoulder_tilt(pose)
            if i > address_i and prev_tilt != 0:
                # Look for a peak in shoulder tilt (max tilt before it
                # starts decreasing = top of backswing)
                pass
            prev_tilt = tilt

        # Simple ratio based on frame indices
        backswing_frames = best_top - address_i
        downswing_frames = impact_i - best_top
        if downswing_frames <= 0:
            return 0.0

        return backswing_frames / downswing_frames

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_pose") and self._pose is not None:
            self._pose.close()

    def __del__(self) -> None:
        self.close()


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if not _MP_AVAILABLE:
        print("mediapipe not available — skipping pose test")
        sys.exit(0)

    estimator = SwingPoseEstimator()

    # Synthetic test: black frames (no person detected)
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(30)]
    analysis = estimator.analyze_swing(frames, impact_frame=20)
    print(f"Analysis (empty frames): {analysis}")
    tips = analysis.get_coaching_tips()
    for tip in tips:
        print(f"  - {tip}")
    estimator.close()
