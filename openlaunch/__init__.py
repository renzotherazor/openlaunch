"""
OpenLaunch — Open-source golf launch monitor and swing coach.

Uses iPhone camera footage + computer vision + ball flight physics
to measure launch metrics and provide coaching feedback.

Usage:
    from openlaunch import LaunchMonitor, SwingCoach, BallFlightSimulator

    # Analyze a swing video
    monitor = LaunchMonitor("swing_240fps.mov", club_length_inches=45.75)
    result = monitor.analyze()

    # Run ball flight simulation
    from openlaunch.physics import BallFlightSimulator, LaunchConditions
    sim = BallFlightSimulator()
    carry = sim.estimate_carry(ball_speed_ms=62, launch_angle_deg=11, spin_rate=2500)
"""

__version__ = "0.1.0"

# Lazy imports — only load heavy modules when needed
def __getattr__(name):
    if name == "LaunchMonitor":
        from openlaunch.launch_monitor import LaunchMonitor
        return LaunchMonitor
    if name == "SwingCoach":
        from openlaunch.swing_coach import SwingCoach
        return SwingCoach
    if name == "BallFlightSimulator":
        from openlaunch.physics import BallFlightSimulator
        return BallFlightSimulator
    if name == "LaunchConditions":
        from openlaunch.physics import LaunchConditions
        return LaunchConditions
    raise AttributeError(f"module 'openlaunch' has no attribute {name!r}")
