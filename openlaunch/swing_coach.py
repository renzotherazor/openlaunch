"""
OpenLaunch Swing Coach — Biomechanical analysis and coaching feedback.

Provides research-based coaching recommendations derived from swing metrics.
References:
  - Cheetham et al. (2001) "The biomechanics of the golf swing"
  - Nesbit & Serrano (2005) "Work and power analysis of the golf swing"
  - TrackMan University data on tour averages
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Club reference data (PGA Tour averages) ────────────────────────

CLUB_SPECS: dict[str, dict] = {
    "driver":   {"loft": 10.5, "length": 45.75, "avg_club_speed_mph": 113, "avg_ball_speed_mph": 167, "avg_smash": 1.48, "avg_launch": 10.9, "avg_spin": 2686},
    "3-wood":   {"loft": 15.0, "length": 43.25, "avg_club_speed_mph": 106, "avg_ball_speed_mph": 155, "avg_smash": 1.46, "avg_launch": 12.2, "avg_spin": 3350},
    "5-wood":   {"loft": 19.0, "length": 42.25, "avg_club_speed_mph": 101, "avg_ball_speed_mph": 146, "avg_smash": 1.45, "avg_launch": 14.3, "avg_spin": 3950},
    "3-iron":   {"loft": 20.0, "length": 39.50, "avg_club_speed_mph": 98,  "avg_ball_speed_mph": 141, "avg_smash": 1.44, "avg_launch": 15.2, "avg_spin": 4100},
    "4-iron":   {"loft": 23.0, "length": 38.50, "avg_club_speed_mph": 95,  "avg_ball_speed_mph": 136, "avg_smash": 1.43, "avg_launch": 16.5, "avg_spin": 4600},
    "5-iron":   {"loft": 26.0, "length": 37.50, "avg_club_speed_mph": 92,  "avg_ball_speed_mph": 131, "avg_smash": 1.42, "avg_launch": 17.8, "avg_spin": 5100},
    "6-iron":   {"loft": 29.0, "length": 37.00, "avg_club_speed_mph": 89,  "avg_ball_speed_mph": 127, "avg_smash": 1.43, "avg_launch": 19.0, "avg_spin": 5700},
    "7-iron":   {"loft": 32.5, "length": 36.50, "avg_club_speed_mph": 86,  "avg_ball_speed_mph": 122, "avg_smash": 1.42, "avg_launch": 20.5, "avg_spin": 6400},
    "8-iron":   {"loft": 36.0, "length": 36.00, "avg_club_speed_mph": 83,  "avg_ball_speed_mph": 117, "avg_smash": 1.41, "avg_launch": 22.0, "avg_spin": 7100},
    "9-iron":   {"loft": 40.0, "length": 35.50, "avg_club_speed_mph": 80,  "avg_ball_speed_mph": 112, "avg_smash": 1.40, "avg_launch": 23.5, "avg_spin": 7800},
    "pw":       {"loft": 45.0, "length": 35.25, "avg_club_speed_mph": 77,  "avg_ball_speed_mph": 107, "avg_smash": 1.39, "avg_launch": 25.0, "avg_spin": 8600},
    "sw":       {"loft": 55.0, "length": 35.00, "avg_club_speed_mph": 70,  "avg_ball_speed_mph": 93,  "avg_smash": 1.33, "avg_launch": 30.0, "avg_spin": 10500},
    "lw":       {"loft": 60.0, "length": 35.00, "avg_club_speed_mph": 66,  "avg_ball_speed_mph": 85,  "avg_smash": 1.29, "avg_launch": 33.0, "avg_spin": 11200},
}


@dataclass
class CoachingTip:
    """A single coaching recommendation."""
    category: str          # e.g. 'power', 'consistency', 'accuracy', 'technique'
    priority: str          # 'high', 'medium', 'low'
    title: str             # short title
    detail: str            # explanation
    drill: str             # suggested drill or exercise
    metric: str            # which metric triggered this
    current: float         # current value
    target: tuple[float, float]  # (min, max) ideal range


@dataclass
class SwingCoachReport:
    """Complete coaching report for a swing analysis."""
    club: str
    tips: list[CoachingTip] = field(default_factory=list)
    power_score: float = 0.0     # 0-100
    consistency_score: float = 0.0  # 0-100
    accuracy_score: float = 0.0   # 0-100
    overall_score: float = 0.0    # 0-100

    def summary(self) -> str:
        """Human-readable coaching summary."""
        lines = [
            f"🏌️  OpenLaunch Swing Coach — {self.club.upper()}",
            f"{'='*50}",
            f"Power:       {self.power_score:.0f}/100",
            f"Consistency: {self.consistency_score:.0f}/100",
            f"Accuracy:    {self.accuracy_score:.0f}/100",
            f"Overall:     {self.overall_score:.0f}/100",
            "",
        ]
        if self.tips:
            lines.append("COACHING TIPS:")
            for i, tip in enumerate(self.tips, 1):
                marker = {"high": "🔴", "medium": "🟡", "low": "🟢"}[tip.priority]
                lines.append(f"  {marker} {i}. {tip.title}")
                lines.append(f"     {tip.detail}")
                lines.append(f"     Drill: {tip.drill}")
                lines.append(f"     {tip.metric}: {tip.current:.1f} (ideal: {tip.target[0]:.1f}-{tip.target[1]:.1f})")
                lines.append("")
        return "\n".join(lines)


class SwingCoach:
    """
    Evaluates swing metrics against research-based benchmarks and
    generates actionable coaching feedback.
    """

    def __init__(self, club: str = "driver"):
        self.club = club.lower()
        self.specs = CLUB_SPECS.get(self.club, CLUB_SPECS["7-iron"])

    def evaluate(
        self,
        ball_speed_mph: float = 0.0,
        club_speed_mph: float = 0.0,
        smash_factor: float = 0.0,
        launch_angle_deg: float = 0.0,
        spin_rate_rpm: float = 0.0,
        launch_direction_deg: float = 0.0,
        carry_distance_yds: float = 0.0,
        # Swing analysis metrics (optional)
        shoulder_tilt_at_address: float = 0.0,
        shoulder_tilt_at_top: float = 0.0,
        hip_rotation_at_impact: float = 0.0,
        weight_shift_ratio: float = 0.0,
        spine_angle: float = 0.0,
        head_stability_inches: float = 0.0,
        swing_tempo: float = 0.0,
    ) -> SwingCoachReport:
        """Generate coaching report from swing metrics."""
        tips: list[CoachingTip] = []

        # ── Power Analysis ──────────────────────────────────────
        power_score = self._score_power(ball_speed_mph, club_speed_mph, smash_factor, tips)

        # ── Consistency Analysis ────────────────────────────────
        consistency_score = self._score_consistency(
            launch_angle_deg, spin_rate_rpm, launch_direction_deg, tips
        )

        # ── Accuracy Analysis ──────────────────────────────────
        accuracy_score = self._score_accuracy(launch_direction_deg, tips)

        # ── Technique Analysis (if swing data available) ───────
        tech_score = self._score_technique(
            shoulder_tilt_at_address, shoulder_tilt_at_top,
            hip_rotation_at_impact, weight_shift_ratio,
            spine_angle, head_stability_inches, swing_tempo, tips,
        )

        # ── Overall Score ──────────────────────────────────────
        n_components = 1
        total = power_score
        if consistency_score > 0:
            total += consistency_score
            n_components += 1
        if accuracy_score > 0:
            total += accuracy_score
            n_components += 1
        if tech_score > 0:
            total += tech_score
            n_components += 1
        overall = total / n_components

        return SwingCoachReport(
            club=self.club,
            tips=tips,
            power_score=power_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            overall_score=overall,
        )

    def _score_power(self, ball_speed, club_speed, smash, tips) -> float:
        """Score power metrics against tour averages."""
        if ball_speed <= 0:
            return 0.0

        spec = self.specs
        # Ball speed as % of tour average
        ball_pct = min(100, (ball_speed / spec["avg_ball_speed_mph"]) * 100) if spec["avg_ball_speed_mph"] else 50
        power_score = ball_pct * 0.7  # 70% weight on ball speed

        # Smash factor check
        if smash > 0:
            ideal_smash = spec["avg_smash"]
            smash_diff = abs(smash - ideal_smash)
            if smash_diff > 0.1:
                tips.append(CoachingTip(
                    category="power",
                    priority="high" if smash < ideal_smash - 0.05 else "medium",
                    title="Smash factor below optimal" if smash < ideal_smash else "Smash factor above expected",
                    detail=(
                        f"Your smash factor ({smash:.2f}) is {'below' if smash < ideal_smash else 'above'} "
                        f"the PGA Tour average ({ideal_smash:.2f}) for {self.club}. "
                        + ("Below-optimal smash suggests off-center contact or excessive spin." if smash < ideal_smash
                           else "Above-expected smash may indicate a very efficient strike — verify with multiple shots.")
                    ),
                    drill="Impact tape drill: Place impact tape on clubface to verify center contact.",
                    metric="smash_factor",
                    current=smash,
                    target=(ideal_smash - 0.05, ideal_smash + 0.05),
                ))
            smash_pct = max(0, 100 - smash_diff * 200)
            power_score += smash_pct * 0.3  # 30% weight on smash
        else:
            power_score = ball_pct

        return min(100, power_score)

    def _score_consistency(self, launch_angle, spin_rate, launch_dir, tips) -> float:
        """Score launch condition consistency."""
        if launch_angle <= 0:
            return 0.0

        spec = self.specs
        score = 50.0  # neutral

        # Launch angle relative to ideal
        ideal_launch = spec["avg_launch"]
        launch_diff = abs(launch_angle - ideal_launch)
        if launch_diff > 5:
            priority = "high" if launch_diff > 8 else "medium"
            tips.append(CoachingTip(
                category="consistency",
                priority=priority,
                title=f"Launch angle {'too high' if launch_angle > ideal_launch else 'too low'}",
                detail=(
                    f"Launch angle of {launch_angle:.1f}° vs. tour average {ideal_launch:.1f}° for {self.club}. "
                    + ("High launch wastes distance by ballooning. Check dynamic loft and angle of attack."
                       if launch_angle > ideal_launch
                       else "Low launch costs carry. You may be de-lofting at impact or hitting down excessively.")
                ),
                drill="Launch control: Practice hitting shots at different trajectory heights intentionally.",
                metric="launch_angle",
                current=launch_angle,
                target=(ideal_launch - 3, ideal_launch + 3),
            ))
            score -= min(30, launch_diff * 3)
        else:
            score += 20

        # Spin rate check
        if spin_rate > 0:
            ideal_spin = spec["avg_spin"]
            spin_diff_pct = abs(spin_rate - ideal_spin) / ideal_spin * 100
            if spin_diff_pct > 20:
                tips.append(CoachingTip(
                    category="consistency",
                    priority="medium",
                    title=f"Spin rate {'too high' if spin_rate > ideal_spin else 'too low'}",
                    detail=(
                        f"Spin rate of {spin_rate:.0f} rpm vs. tour average {ideal_spin:.0f} rpm for {self.club}. "
                        + ("High spin causes ballooning and loss of distance."
                           if spin_rate > ideal_spin
                           else "Low spin reduces carry and control.")
                    ),
                    drill="Spin control: Check your angle of attack and dynamic loft with a launch monitor.",
                    metric="spin_rate",
                    current=spin_rate,
                    target=(ideal_spin * 0.8, ideal_spin * 1.2),
                ))
                score -= min(20, spin_diff_pct * 0.5)
            else:
                score += 15

        # Launch direction
        if abs(launch_dir) > 5:
            tips.append(CoachingTip(
                category="accuracy",
                priority="high",
                title=f"Ball starting {'right' if launch_dir > 0 else 'left'} of target",
                detail=f"Launch direction {launch_dir:.1f}° off target line. Face controls ~75% of starting direction.",
                drill="Alignment sticks: Place alignment sticks on ground pointing at target during practice.",
                metric="launch_direction",
                current=launch_dir,
                target=(-3, 3),
            ))

        return max(0, min(100, score))

    def _score_accuracy(self, launch_dir, tips) -> float:
        """Score directional accuracy."""
        if abs(launch_dir) < 0.1:
            return 50.0  # no data, neutral
        # Within 3° is excellent, 5° acceptable, >7° needs work
        if abs(launch_dir) <= 3:
            return 95.0
        elif abs(launch_dir) <= 5:
            return 75.0
        elif abs(launch_dir) <= 7:
            return 55.0
        else:
            return 30.0

    def _score_technique(
        self, shoulder_addr, shoulder_top, hip_rot, weight_shift,
        spine, head_stab, tempo, tips
    ) -> float:
        """Score swing technique metrics (returns 0 if no data)."""
        has_data = any(v > 0 for v in [shoulder_addr, shoulder_top, hip_rot, weight_shift, spine, head_stab, tempo])
        if not has_data:
            return 0.0

        score = 50.0

        # Shoulder tilt at top (for right-handed: rightward tilt)
        if shoulder_top > 0:
            if 30 <= shoulder_top <= 45:
                score += 12
            elif shoulder_top < 25:
                tips.append(CoachingTip(
                    category="technique", priority="medium",
                    title="Insufficient shoulder tilt at top",
                    detail=f"Shoulder tilt of {shoulder_top:.1f}° at top — ideal is 30-45°. Shallow turn limits power and consistency.",
                    drill="Mirror drill: Practice backswing in front of a mirror, focus on shoulder over back foot.",
                    metric="shoulder_tilt_at_top", current=shoulder_top, target=(30, 45),
                ))
                score -= 8
            elif shoulder_top > 50:
                tips.append(CoachingTip(
                    category="technique", priority="low",
                    title="Excessive shoulder tilt",
                    detail=f"Shoulder tilt of {shoulder_top:.1f}° — over-tilting can cause reverse pivot and balance issues.",
                    drill="Weight distribution check: Ensure weight stays on inside of back foot at top.",
                    metric="shoulder_tilt_at_top", current=shoulder_top, target=(30, 45),
                ))

        # Hip rotation at impact
        if hip_rot > 0:
            if 35 <= hip_rot <= 50:
                score += 10
            elif hip_rot < 30:
                tips.append(CoachingTip(
                    category="technique", priority="high",
                    title="Hip rotation at impact too slow",
                    detail=f"Hip rotation of {hip_rot:.1f}° at impact — ideal 35-50°. Sliding instead of rotating costs power and accuracy.",
                    drill="Step drill: Take practice swings with a step forward to feel hip rotation leading the downswing.",
                    metric="hip_rotation_at_impact", current=hip_rot, target=(35, 50),
                ))
                score -= 10

        # Swing tempo
        if tempo > 0:
            if 2.5 <= tempo <= 3.5:
                score += 8
            else:
                tips.append(CoachingTip(
                    category="technique",
                    priority="medium" if tempo < 2.0 else "low",
                    title=f"Swing tempo {'too fast' if tempo < 2.5 else 'too slow'} on downswing",
                    detail=f"Tempo ratio {tempo:.1f}:1 (backswing:downswing) — ideal 2.5-3.5:1. "
                           + ("Rushing downswing causes poor sequencing and inconsistent contact."
                              if tempo < 2.5
                              else "Very slow downswing may indicate deceleration or over-control."),
                    drill="Metronome drill: Use a golf metronome app set to your desired tempo ratio.",
                    metric="swing_tempo", current=tempo, target=(2.5, 3.5),
                ))

        # Head stability
        if head_stab > 0:
            if head_stab <= 2:
                score += 8
            elif head_stab > 3:
                tips.append(CoachingTip(
                    category="technique", priority="medium",
                    title="Excessive head movement during swing",
                    detail=f"Head moved {head_stab:.1f}\" laterally — ideal < 2\". Excessive movement affects consistent ball striking.",
                    drill="Wall drill: Stand with head near a wall, practice keeping head still through the swing.",
                    metric="head_stability", current=head_stab, target=(0, 2),
                ))

        return max(0, min(100, score))


if __name__ == "__main__":
    # Demo with PGA-tour-level input
    coach = SwingCoach("driver")
    report = coach.evaluate(
        ball_speed_mph=165,
        club_speed_mph=112,
        smash_factor=1.47,
        launch_angle_deg=11.5,
        spin_rate_rpm=2600,
        launch_direction_deg=-1.2,
        shoulder_tilt_at_top=38,
        hip_rotation_at_impact=42,
        swing_tempo=3.0,
        head_stability_inches=1.5,
    )
    print(report.summary())

    # Demo with amateur input
    print("\n" + "=" * 50 + "\n")
    coach2 = SwingCoach("7-iron")
    report2 = coach2.evaluate(
        ball_speed_mph=105,
        club_speed_mph=78,
        smash_factor=1.35,
        launch_angle_deg=24,
        spin_rate_rpm=8200,
        launch_direction_deg=6.5,
        shoulder_tilt_at_top=22,
        hip_rotation_at_impact=25,
        swing_tempo=1.8,
        head_stability_inches=4.0,
    )
    print(report2.summary())
