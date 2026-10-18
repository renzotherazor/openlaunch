"""Club-length and ball-diameter calibration for pixel ↔ real-world conversion.

Two calibration strategies are supported:
1. **Club-length**: user marks the top of the grip and the sole of the club.
2. **Ball-diameter**: user marks the ball center and edge (diameter = 1.68 in).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Standard golf ball diameter in inches (USGA rule)
GOLF_BALL_DIAMETER_INCHES: float = 1.68


class Calibration:
    """Pixel ↔ real-world unit conversion for a given camera setup."""

    def __init__(
        self,
        club_length_inches: float = 45.75,
        camera_angle: str = "face_on",
    ) -> None:
        if club_length_inches <= 0:
            raise ValueError("club_length_inches must be positive")
        if camera_angle not in ("face_on", "down_the_line"):
            raise ValueError("camera_angle must be 'face_on' or 'down_the_line'")

        self.club_length_inches: float = club_length_inches
        self.camera_angle: str = camera_angle
        self.pixels_per_inch: float = 0.0  # set during calibration

    # ------------------------------------------------------------------
    # Calibration methods
    # ------------------------------------------------------------------

    def calibrate_from_frame(
        self,
        frame: np.ndarray,
        top_point: tuple[int, int],
        sole_point: tuple[int, int],
    ) -> float:
        """Calibrate using the club length as reference.

        User marks the top of the grip and the sole of the clubhead.
        The distance in pixels between these two points corresponds to
        ``club_length_inches``.

        Parameters
        ----------
        frame : BGR image (used only for validation of point bounds).
        top_point : (x, y) pixel coordinates of grip top.
        sole_point : (x, y) pixel coordinates of club sole.

        Returns
        -------
        float : pixels_per_inch
        """
        h, w = frame.shape[:2]
        for pt in (top_point, sole_point):
            if not (0 <= pt[0] < w and 0 <= pt[1] < h):
                raise ValueError(f"Point {pt} outside frame bounds ({w}x{h})")

        pixel_dist = math.hypot(
            top_point[0] - sole_point[0], top_point[1] - sole_point[1]
        )
        if pixel_dist < 1:
            raise ValueError("Top and sole points are too close")

        self.pixels_per_inch = pixel_dist / self.club_length_inches
        return self.pixels_per_inch

    def calibrate_from_ball(
        self,
        frame: np.ndarray,
        ball_center: tuple[int, int],
        ball_edge: tuple[int, int],
        ball_diameter_inches: float = GOLF_BALL_DIAMETER_INCHES,
    ) -> float:
        """Calibrate using the golf ball diameter as reference.

        User marks the ball center and any point on the ball edge.  The
        distance from center to edge is the radius; diameter = 2 × radius.

        Parameters
        ----------
        frame : BGR image.
        ball_center : (x, y) pixel coordinates of ball center.
        ball_edge : (x, y) pixel coordinates of ball edge.
        ball_diameter_inches : real-world ball diameter (default 1.68").

        Returns
        -------
        float : pixels_per_inch
        """
        h, w = frame.shape[:2]
        for pt in (ball_center, ball_edge):
            if not (0 <= pt[0] < w and 0 <= pt[1] < h):
                raise ValueError(f"Point {pt} outside frame bounds ({w}x{h})")

        pixel_radius = math.hypot(
            ball_center[0] - ball_edge[0], ball_center[1] - ball_edge[1]
        )
        if pixel_radius < 1:
            raise ValueError("Center and edge points are too close")

        pixel_diameter = 2.0 * pixel_radius
        self.pixels_per_inch = pixel_diameter / ball_diameter_inches
        return self.pixels_per_inch

    # ------------------------------------------------------------------
    # Auto-calibration helper
    # ------------------------------------------------------------------

    def auto_calibrate_from_ball(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> float | None:
        """Attempt automatic calibration using HoughCircles to find a golf ball.

        Returns pixels_per_inch if a ball is found, else None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y : y + h, x : x + w]

        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=80,
            param2=30,
            minRadius=8,
            maxRadius=40,
        )

        if circles is None or len(circles) == 0:
            return None

        # Use the largest circle (most likely a ball vs. noise)
        best = max(circles[0], key=lambda c: c[2])
        pixel_radius = best[2]
        if pixel_radius < 3:
            return None

        pixel_diameter = 2.0 * pixel_radius
        self.pixels_per_inch = pixel_diameter / GOLF_BALL_DIAMETER_INCHES
        return self.pixels_per_inch

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def pixels_to_real(self, pixels: float) -> float:
        """Convert pixels → inches."""
        if self.pixels_per_inch <= 0:
            raise RuntimeError("Calibration not yet performed (pixels_per_inch == 0)")
        return pixels / self.pixels_per_inch

    def pixels_to_real_m(self, pixels: float) -> float:
        """Convert pixels → meters."""
        return self.pixels_to_real(pixels) * 0.0254

    def real_to_pixels(self, inches: float) -> float:
        """Convert inches → pixels."""
        if self.pixels_per_inch <= 0:
            raise RuntimeError("Calibration not yet performed (pixels_per_inch == 0)")
        return inches * self.pixels_per_inch

    @property
    def is_calibrated(self) -> bool:
        return self.pixels_per_inch > 0

    def __repr__(self) -> str:
        return (
            f"Calibration(ppi={self.pixels_per_inch:.2f}, "
            f"club={self.club_length_inches:.2f}in, "
            f"angle={self.camera_angle})"
        )


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    # Synthetic test: 1920x1080 frame, club occupies 400 pixels vertically
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cal = Calibration(club_length_inches=45.75)
    ppi = cal.calibrate_from_frame(frame, top_point=(960, 100), sole_point=(960, 500))
    print(f"Calibrated: {ppi:.2f} px/in")
    print(f"400 px = {cal.pixels_to_real(400):.2f} in")
    print(f"45.75 in = {cal.real_to_pixels(45.75):.2f} px")

    # Ball calibration
    cal2 = Calibration()
    ppi2 = cal2.calibrate_from_ball(frame, ball_center=(500, 540), ball_edge=(517, 540))
    print(f"\nBall calibration: {ppi2:.2f} px/in")
    print(f"1.68 in (ball) = {cal2.real_to_pixels(1.68):.2f} px")
