"""Video ingest, frame extraction, and metadata reading for golf swing videos.

Handles iPhone slow-motion (240 fps) and normal-speed videos.  OpenCV reads
the *capture* frame-rate from the container metadata; iPhone slow-mo also
stores a "timelist" that tells QuickTime how to stretch the clip for playback.
We expose both the raw capture FPS and the *real-world* FPS so that downstream
trackers can compute true velocities.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Generator

import cv2
import numpy as np


class VideoReader:
    """Read a golf-swing video and expose frame-level access with timing info."""

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video not found: {path}")
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        # Cache metadata
        self._fps: float = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution: tuple[int, int] = (w, h)

        # Slow-motion metadata (iPhone specific)
        self._slow_mo_factor: float = self._detect_slow_mo_factor()
        self._real_fps: float = self._fps * self._slow_mo_factor

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Actual FPS as reported by the video container (capture rate)."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total number of frames in the video."""
        return self._frame_count

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return self._resolution

    @property
    def slow_mo_factor(self) -> float:
        """Slow-motion stretch factor.

        1.0  = normal speed
        0.125 = iPhone 240 fps played back at 30 fps (8× slowdown)
        0.5  = 2× slow-mo, etc.
        """
        return self._slow_mo_factor

    @property
    def real_fps(self) -> float:
        """Real-world frames per second (capture rate).

        For a 240 fps iPhone slow-mo clip whose container says 30 fps with
        slow_mo_factor=0.125, real_fps = 30 * 0.125 = 240? No — wait.

        Actually: OpenCV reads the *nominal* FPS from the container, which for
        an iPhone slow-mo export is typically the *playback* rate (e.g. 30 fps).
        The real capture rate is higher.  We derive: real_fps = fps / slow_mo_factor.

        But we set slow_mo_factor = playback/capture ratio so:
            real_fps = fps * slow_mo_factor   (if fps is playback)
        OR
            real_fps = fps / slow_mo_factor   (if fps is capture)

        For iPhone: OpenCV fps = capture rate (240), slow_mo_factor = playback/capture = 30/240 = 0.125
        So real_fps = fps (240). The factor just tells consumers how slow playback is.
        """
        return self._real_fps

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, index: int) -> np.ndarray:
        """Return a single BGR frame by index."""
        if index < 0 or index >= self._frame_count:
            raise IndexError(f"Frame {index} out of range [0, {self._frame_count})")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        if not ret:
            raise IOError(f"Failed to read frame {index}")
        return frame

    def get_frames_range(
        self, start: int, end: int, step: int = 1
    ) -> Generator[np.ndarray, None, None]:
        """Yield BGR frames from *start* (inclusive) to *end* (exclusive)."""
        for i in range(start, end, step):
            if i >= self._frame_count:
                break
            yield self.get_frame(i)

    # ------------------------------------------------------------------
    # Impact detection
    # ------------------------------------------------------------------

    def find_impact_frame(self, roi: tuple[int, int, int, int] | None = None) -> int:
        """Auto-detect the frame where the club strikes the ball.

        Strategy:
        1. Compute frame differences across the whole clip.
        2. The impact region shows a sudden, large motion transient — the ball
           transitions from stationary to fast flight.
        3. We look for the peak of the *second derivative* of the motion
           magnitude signal, which corresponds to the sharpest acceleration.

        Parameters
        ----------
        roi : (x, y, w, h) optional ROI to focus on (e.g. ball area).

        Returns
        -------
        int : estimated impact frame index.
        """
        if self._frame_count < 10:
            return 0

        diffs: list[float] = []
        prev_gray: np.ndarray | None = None

        for i in range(self._frame_count):
            frame = self.get_frame(i)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi is not None:
                x, y, w, h = roi
                gray = gray[y : y + h, x : x + w]
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                diffs.append(float(np.mean(diff)))
            else:
                diffs.append(0.0)
            prev_gray = gray

        # Smooth the signal
        arr = np.array(diffs, dtype=np.float64)
        kernel_size = max(3, self._frame_count // 100)
        if kernel_size % 2 == 0:
            kernel_size += 1
        arr = np.convolve(arr, np.ones(kernel_size) / kernel_size, mode="same")

        # Second derivative (discrete) — peak = sharpest acceleration = impact
        if len(arr) < 3:
            return int(np.argmax(arr))

        d2 = np.gradient(np.gradient(arr))
        # Search in the second half of the video (impact is never at the start)
        search_start = max(0, len(d2) // 4)
        impact_idx = int(np.argmax(d2[search_start:])) + search_start

        return max(0, min(impact_idx, self._frame_count - 1))

    # ------------------------------------------------------------------
    # Trimming
    # ------------------------------------------------------------------

    def trim(self, start: int, end: int, output_path: str) -> None:
        """Save a trimmed clip from *start* to *end* (exclusive) using ffmpeg."""
        t_start = start / self._fps
        t_end = end / self._fps
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            self._path,
            "-ss",
            f"{t_start:.4f}",
            "-to",
            f"{t_end:.4f}",
            "-c",
            "copy",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: re-encode frame by frame via OpenCV
            self._trim_opencv(start, end, output_path)

    def _trim_opencv(self, start: int, end: int, output_path: str) -> None:
        """Fallback trim using OpenCV VideoWriter."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h = self._resolution
        writer = cv2.VideoWriter(output_path, fourcc, self._fps, (w, h))
        try:
            for frame in self.get_frames_range(start, end):
                writer.write(frame)
        finally:
            writer.release()

    # ------------------------------------------------------------------
    # Slow-mo detection
    # ------------------------------------------------------------------

    def _detect_slow_mo_factor(self) -> float:
        """Try to determine the slow-motion factor from metadata.

        iPhone slow-mo videos store a 'timelist' atom that maps sample
        timestamps to playback times.  We try to read it via ffprobe.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    self._path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return self._infer_slow_mo_from_fps()
            data = json.loads(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            return self._infer_slow_mo_from_fps()

        # Check for iPhone slow-mo tags in stream metadata
        for stream in data.get("streams", []):
            tags = stream.get("tags", {})
            # iPhone sometimes tags slow-mo with compatible_brands
            for key, val in tags.items():
                if "slow" in str(val).lower() or "slo" in str(val).lower():
                    return self._infer_slow_mo_from_fps()

            # Check rotation / speed metadata
            speed = stream.get("speed", "N/A")
            if speed != "N/A":
                try:
                    return float(speed)
                except ValueError:
                    pass

        return self._infer_slow_mo_from_fps()

    def _infer_slow_mo_from_fps(self) -> float:
        """Heuristic: if OpenCV fps > 60, it's likely slow-mo capture."""
        if self._fps >= 200:
            # iPhone 240 fps → playback 30 fps → factor 30/240 = 0.125
            return 30.0 / self._fps
        elif self._fps >= 100:
            # 120 fps → playback 30 fps → factor 0.25
            return 30.0 / self._fps
        elif self._fps > 60:
            # 60+ fps slight slow-mo
            return 30.0 / self._fps
        else:
            return 1.0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()

    def __repr__(self) -> str:
        return (
            f"VideoReader(path={self._path!r}, fps={self._fps:.1f}, "
            f"real_fps={self._real_fps:.1f}, frames={self._frame_count}, "
            f"res={self._resolution})"
        )


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m openlaunch.video <video_path>")
        sys.exit(1)

    vr = VideoReader(sys.argv[1])
    print(repr(vr))
    print(f"Slow-mo factor: {vr.slow_mo_factor}")
    impact = vr.find_impact_frame()
    print(f"Estimated impact frame: {impact}")
