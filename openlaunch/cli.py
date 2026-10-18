"""
OpenLaunch CLI — Command-line interface for the golf launch monitor.

Usage:
    openlaunch analyze swing.mov --club driver --club-length 45.75
    openlaunch simulate --ball-speed 150 --launch-angle 11 --spin 2500
    openlaunch clubs                     # Show reference club data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_analyze(args):
    """Analyze a golf swing video."""
    from openlaunch.launch_monitor import LaunchMonitor

    monitor = LaunchMonitor(
        video_path=args.video,
        club_length_inches=args.club_length,
        camera_angle=args.angle,
        real_fps=args.real_fps,
    )
    result = monitor.analyze(
        impact_frame=args.impact_frame,
        calibration_points=args.calibration,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())

    # Run coaching if swing analysis available
    if result.swing_analysis is not None:
        from openlaunch.swing_coach import SwingCoach
        coach = SwingCoach(args.club)
        report = coach.evaluate(
            ball_speed_mph=result.ball_speed_mph,
            club_speed_mph=result.club_speed_mph,
            smash_factor=result.smash_factor,
            launch_angle_deg=result.launch_angle_deg,
            spin_rate_rpm=result.spin_rate_rpm,
            launch_direction_deg=result.launch_direction_deg,
        )
        print("\n" + report.summary())

    # Generate trajectory chart if requested
    if args.chart and result.trajectory_points:
        from openlaunch.physics import BallFlightSimulator
        sim = BallFlightSimulator()
        # trajectory_points may be a list of (x_m, z_m, t_s) tuples
        # (from fallback path) or a proper TrajectoryResult
        tp = result.trajectory_points
        if hasattr(tp, 'points'):
            sim.to_trajectory_chart(tp, filename=args.chart)
        else:
            # Convert list of tuples to a chart manually
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            x_yards = [p[0] * 1.09361 for p in tp]
            z_m = [p[1] for p in tp]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_yards, z_m, color="#1f77b4", linewidth=2)
            ax.set_xlabel("Distance (yards)")
            ax.set_ylabel("Height (m)")
            ax.set_title("Ball Flight Trajectory")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            if args.chart.endswith(".png"):
                fig.savefig(args.chart, dpi=150, bbox_inches="tight")
            else:
                fig.savefig(args.chart, bbox_inches="tight")
            plt.close(fig)
        print(f"\nTrajectory chart saved to: {args.chart}")


def cmd_simulate(args):
    """Simulate ball flight from manual launch conditions."""
    from openlaunch.physics import BallFlightSimulator, LaunchConditions
    import math

    sim = BallFlightSimulator()
    launch = LaunchConditions(
        velocity=args.ball_speed / 2.237 if args.mph else args.ball_speed,
        launch_angle=math.radians(args.launch_angle),
        azimuth_angle=math.radians(args.azimuth or 0),
        spin_rate=args.spin,
    )
    trajectory = sim.simulate_trajectory(launch)

    print(f"\n🏌️  Launch Simulation Results")
    print(f"{'='*40}")
    print(f"Ball Speed:       {args.ball_speed:.1f} {'mph' if args.mph else 'm/s'}")
    print(f"Launch Angle:     {args.launch_angle:.1f}°")
    print(f"Spin Rate:        {args.spin:.0f} rpm")
    print(f"{'─'*40}")
    print(f"Carry Distance:   {trajectory.carry_distance * 1.09361:.1f} yds")
    print(f"Max Height:       {trajectory.max_height:.1f} m ({trajectory.max_height * 3.281:.1f} ft)")
    print(f"Flight Time:      {trajectory.flight_time:.2f} s")
    print(f"Landing Angle:    {trajectory.descent_angle:.1f}°")
    print(f"Landing Speed:    {trajectory.landing_velocity * 2.237:.1f} mph")

    if args.chart:
        sim.to_trajectory_chart(trajectory, filename=args.chart)
        print(f"\nTrajectory chart saved to: {args.chart}")
    else:
        # Show inline SVG
        svg = sim.to_trajectory_chart(trajectory)
        print(f"\n(SVG chart generated — {len(svg)} bytes. Use --chart to save.)")


def cmd_clubs(args):
    """Display club reference data."""
    from openlaunch.swing_coach import CLUB_SPECS

    print(f"\n🏌️  Club Reference Data (PGA Tour Averages)")
    print(f"{'='*75}")
    print(f"{'Club':<10} {'Loft':>5} {'Length':>7} {'Club Spd':>9} {'Ball Spd':>9} {'Smash':>6} {'Launch':>7} {'Spin':>6}")
    print(f"{'─'*75}")
    for name, spec in CLUB_SPECS.items():
        print(f"{name:<10} {spec['loft']:>5.1f}° {spec['length']:>6.2f}\" "
              f"{spec['avg_club_speed_mph']:>8.0f}mph {spec['avg_ball_speed_mph']:>8.0f}mph "
              f"{spec['avg_smash']:>6.2f} {spec['avg_launch']:>6.1f}° {spec['avg_spin']:>5.0f}")


def main():
    parser = argparse.ArgumentParser(
        prog="openlaunch",
        description="OpenLaunch — Open-source golf launch monitor and swing coach",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── analyze ───────────────────────────────────────────────
    p_analyze = subparsers.add_parser("analyze", help="Analyze a golf swing video")
    p_analyze.add_argument("video", help="Path to swing video (iPhone slow-mo recommended)")
    p_analyze.add_argument("--club", default="driver", help="Club name (driver, 7-iron, pw, etc.)")
    p_analyze.add_argument("--club-length", type=float, default=45.75, help="Club length in inches")
    p_analyze.add_argument("--angle", choices=["face_on", "down_the_line"], default="face_on",
                           help="Camera angle")
    p_analyze.add_argument("--impact-frame", type=int, default=None,
                           help="Frame index of impact (auto-detected if not specified)")
    p_analyze.add_argument("--calibration", type=str, default=None,
                           help="Calibration points as 'top_x,top_y,sole_x,sole_y'")
    p_analyze.add_argument("--chart", type=str, default=None,
                           help="Save trajectory chart to file (SVG or PNG)")
    p_analyze.add_argument("--json", action="store_true", help="Output as JSON")
    p_analyze.add_argument("--real-fps", type=float, default=None,
                           help="True capture FPS for baked slow-motion files "
                                "(e.g. 240 for a 240fps clip exported at 30fps). "
                                "Use when video metadata reports wrong fps.")
    p_analyze.set_defaults(func=cmd_analyze)

    # ── simulate ───────────────────────────────────────────────
    p_sim = subparsers.add_parser("simulate", help="Simulate ball flight from launch conditions")
    p_sim.add_argument("--ball-speed", type=float, required=True, help="Ball speed (mph or m/s)")
    p_sim.add_argument("--mph", action="store_true", default=True, help="Ball speed in mph (default)")
    p_sim.add_argument("--launch-angle", type=float, required=True, help="Launch angle in degrees")
    p_sim.add_argument("--spin", type=float, required=True, help="Backspin rate in rpm")
    p_sim.add_argument("--azimuth", type=float, default=0, help="Horizontal launch direction in degrees")
    p_sim.add_argument("--chart", type=str, default=None, help="Save trajectory chart")
    p_sim.set_defaults(func=cmd_simulate)

    # ── clubs ──────────────────────────────────────────────────
    p_clubs = subparsers.add_parser("clubs", help="Show club reference data")
    p_clubs.set_defaults(func=cmd_clubs)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
