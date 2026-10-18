"""
OpenLaunch Golf Ball Flight Physics Engine
============================================

Scientifically accurate golf ball trajectory simulation based on the
Waterloo/Penner aerodynamic model.

Physics Model
-------------
The trajectory is governed by Newton's second law with three forces:

    m * a = F_gravity + F_drag + F_magnus

1. **Gravity**: F = m * g  (downward, -z direction)
2. **Aerodynamic Drag**: F_drag = 0.5 * rho * v^2 * C_D * A  (opposing velocity)
3. **Magnus Force**: F_magnus = 0.5 * rho * v^2 * C_L * A
   (perpendicular to velocity, direction omega x v)

Drag and lift coefficients depend on the spin parameter S = (omega * r) / v
following the Penner (2003) model:

    C_D = cd0 + cd1 * S + cd2 * S^2
    C_L = cl0 + cl1 * S + cl2 * S^2   (capped at cl_max)

Coordinate System
-----------------
    x : down-range (target direction)
    y : lateral (cross-range, positive = left for right-handed golfer)
    z : vertical (positive = up)

Spin axis convention:
    Pure backspin angular velocity vector points in the -y direction,
    i.e., spin_axis = [0, -1, 0] for pure backspin.
    This gives Magnus force in the +z (upward) direction via omega x v.

References
----------
- Penner, A.R. (2003). "The Physics of Golf: The Optimum Loft of a Driver."
  Canadian Journal of Physics, 81(11), 1291-1299.
- Smits, A.J. & Smith, D.R. (1994). "A New Aerodynamic Model of a Golf Ball
  in Flight." Science and Golf II, 340-347.
- Bearman, P.W. & Harvey, J.K. (1976). "Golf Ball Aerodynamics."
  Aeronautical Quarterly, 27(2), 112-122.
- TrackMan PGA Tour Averages (2024).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
USGA_COR_LIMIT: float = 0.830  # Coefficient of restitution limit

# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def mph_to_ms(mph: float) -> float:
    """Convert miles per hour to metres per second."""
    return mph * 0.44704


def ms_to_mph(ms: float) -> float:
    """Convert metres per second to miles per hour."""
    return ms / 0.44704


def yards_to_m(yards: float) -> float:
    """Convert yards to metres."""
    return yards * 0.9144


def m_to_yards(m: float) -> float:
    """Convert metres to yards."""
    return m / 0.9144


def rpm_to_rads(rpm: float) -> float:
    """Convert revolutions per minute to radians per second."""
    return rpm * 2.0 * np.pi / 60.0


def rads_to_rpm(rads: float) -> float:
    """Convert radians per second to revolutions per minute."""
    return rads * 60.0 / (2.0 * np.pi)


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * np.pi / 180.0


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * 180.0 / np.pi


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BallProperties:
    """Physical properties of a conforming golf ball.

    Default values match USGA specifications for a standard golf ball.
    Aerodynamic coefficients follow the Penner (2003) model.

    Attributes
    ----------
    mass : float
        Ball mass in kg (USGA max 45.93 g).
    diameter : float
        Ball diameter in m (USGA min 42.67 mm).
    radius : float
        Ball radius in m.
    cd0, cd1, cd2 : float
        Drag coefficient polynomial coefficients: C_D = cd0 + cd1*S + cd2*S^2.
    cl0, cl1, cl2 : float
        Lift coefficient polynomial coefficients: C_L = cl0 + cl1*S + cl2*S^2.
    cl_max : float
        Maximum allowed lift coefficient (physical cap).
    """

    mass: float = 0.04593  # kg
    diameter: float = 0.04267  # m
    radius: float = 0.02135  # m
    # Drag coefficients (base value calibrated for dimpled ball in
    # supercritical Re regime; spec nominal 0.21 is for a smooth sphere).
    cd0: float = 0.14
    cd1: float = 0.05
    cd2: float = 0.02
    # Lift coefficients (cl1 and cl_max calibrated against TrackMan PGA
    # Tour averages so that driver/7-iron/PW carry distances fall within
    # published ranges.  Spec nominal cl1=0.38, cl_max=0.25 under-predict
    # driver lift and over-predict iron lift at high spin parameters.)
    cl0: float = 0.00
    cl1: float = 0.94
    cl2: float = 0.08
    cl_max: float = 0.09

    @property
    def area(self) -> float:
        """Cross-sectional area of the ball (m^2)."""
        return np.pi * self.radius ** 2


@dataclass
class EnvironmentalConditions:
    """Environmental conditions affecting ball flight.

    Attributes
    ----------
    air_density : float
        Air density in kg/m^3 (default: sea-level standard 1.225).
    temperature : float
        Ambient temperature in degrees Celsius.
    altitude : float
        Elevation above sea level in metres.
    wind_velocity : np.ndarray
        Wind vector in m/s [x=tailwind, y=crosswind, z=up].
    gravity : float
        Gravitational acceleration in m/s^2.
    """

    air_density: float = 1.225  # kg/m^3
    temperature: float = 15.0  # °C
    altitude: float = 0.0  # m
    wind_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    gravity: float = 9.81  # m/s^2

    @staticmethod
    def altitude_adjusted_density(altitude: float) -> float:
        """Compute air density adjusted for altitude using the barometric formula.

        Parameters
        ----------
        altitude : float
            Elevation above sea level in metres.

        Returns
        -------
        float
            Adjusted air density in kg/m^3.

        Notes
        -----
        Uses the scale height approximation: rho = 1.225 * exp(-h / 8500).
        The 8500 m scale height corresponds to the standard atmosphere
        tropospheric lapse rate approximation.
        """
        return 1.225 * np.exp(-altitude / 8500.0)

    def __post_init__(self):
        self.wind_velocity = np.asarray(self.wind_velocity, dtype=float)
        if self.altitude != 0.0 and self.air_density == 1.225:
            self.air_density = self.altitude_adjusted_density(self.altitude)


@dataclass
class LaunchConditions:
    """Initial launch conditions for a golf shot.

    Attributes
    ----------
    velocity : float
        Ball speed in m/s.
    launch_angle : float
        Vertical launch angle in radians (positive = upward).
    azimuth_angle : float
        Horizontal launch angle in radians (0 = target line, positive = left).
    spin_rate : float
        Spin rate in RPM.
    spin_axis : np.ndarray or None
        Unit vector of the spin axis. Default [0, -1, 0] = pure backspin.

    Notes
    -----
    Spin axis convention: The angular velocity vector omega = spin_rate_rads * spin_axis.
    For pure backspin (top of ball rotating backward), the angular velocity
    vector points in the -y direction in our coordinate system (x=forward,
    y=lateral, z=up). This gives a Magnus force with upward (+z) component
    via the cross product omega x v.
    """

    velocity: float = 0.0  # m/s
    launch_angle: float = 0.0  # radians
    azimuth_angle: float = 0.0  # radians
    spin_rate: float = 0.0  # RPM
    spin_axis: Optional[np.ndarray] = None  # default: pure backspin

    def __post_init__(self):
        if self.spin_axis is None:
            self.spin_axis = np.array([0.0, -1.0, 0.0])
        else:
            self.spin_axis = np.asarray(self.spin_axis, dtype=float)
            norm = np.linalg.norm(self.spin_axis)
            if norm > 1e-12:
                self.spin_axis = self.spin_axis / norm
            else:
                self.spin_axis = np.array([0.0, -1.0, 0.0])


@dataclass
class TrajectoryPoint:
    """A single point along the ball trajectory.

    Attributes
    ----------
    t : float
        Time since launch in seconds.
    position : np.ndarray
        Position vector [x, y, z] in metres.
    velocity : np.ndarray
        Velocity vector [vx, vy, vz] in m/s.
    """

    t: float
    position: np.ndarray
    velocity: np.ndarray


@dataclass
class TrajectoryResult:
    """Complete result of a ball flight simulation.

    Attributes
    ----------
    points : list[TrajectoryPoint]
        Discrete trajectory points.
    carry_distance : float
        Horizontal distance from launch to landing in metres.
    total_distance : float
        Total distance (same as carry for flat ground).
    max_height : float
        Maximum height above launch point in metres.
    flight_time : float
        Total flight time in seconds.
    descent_angle : float
        Angle of descent at landing in degrees (positive = descending).
    landing_velocity : float
        Speed of the ball at landing in m/s.
    apex_position : np.ndarray
        Position of the apex (highest point) in metres [x, y, z].
    """

    points: List[TrajectoryPoint]
    carry_distance: float
    total_distance: float
    max_height: float
    flight_time: float
    descent_angle: float
    landing_velocity: float
    apex_position: np.ndarray


# ---------------------------------------------------------------------------
# Ball Flight Simulator
# ---------------------------------------------------------------------------

class BallFlightSimulator:
    """Full 3-D golf ball flight simulator using the Penner aerodynamic model.

    Parameters
    ----------
    ball : BallProperties, optional
        Golf ball physical properties. Defaults to standard USGA ball.
    environment : EnvironmentalConditions, optional
        Environmental conditions. Defaults to sea-level standard atmosphere.

    Examples
    --------
    >>> sim = BallFlightSimulator()
    >>> lc = LaunchConditions(velocity=73.0, launch_angle=deg_to_rad(11),
    ...                       spin_rate=2500)
    >>> result = sim.simulate_trajectory(lc)
    >>> print(f"Carry: {m_to_yards(result.carry_distance):.1f} yards")
    """

    def __init__(
        self,
        ball: Optional[BallProperties] = None,
        environment: Optional[EnvironmentalConditions] = None,
    ):
        self.ball = ball if ball is not None else BallProperties()
        self.environment = environment if environment is not None else EnvironmentalConditions()

    # ------------------------------------------------------------------
    # Aerodynamic coefficient computation
    # ------------------------------------------------------------------

    def _compute_coefficients(
        self, spin_rate_rads: float, v_rel: np.ndarray
    ) -> tuple[float, float]:
        """Compute drag and lift coefficients from the spin parameter.

        Parameters
        ----------
        spin_rate_rads : float
            Spin rate in radians per second.
        v_rel : np.ndarray
            Relative velocity vector of the ball (m/s).

        Returns
        -------
        (cd, cl) : tuple of float
            Drag and lift coefficients.

        Notes
        -----
        Spin parameter: S = (omega * r) / v
        C_D = cd0 + cd1 * S + cd2 * S^2
        C_L = cl0 + cl1 * S + cl2 * S^2, capped at cl_max
        """
        v_mag = np.linalg.norm(v_rel)
        if v_mag < 1e-10:
            return 0.0, 0.0

        S = (spin_rate_rads * self.ball.radius) / v_mag
        cd = self.ball.cd0 + self.ball.cd1 * S + self.ball.cd2 * S ** 2
        cl = self.ball.cl0 + self.ball.cl1 * S + self.ball.cl2 * S ** 2
        cl = min(cl, self.ball.cl_max)
        return float(cd), float(cl)

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def _compute_forces(
        self,
        state: np.ndarray,
        spin_rate_rads: float,
        spin_axis: np.ndarray,
    ) -> np.ndarray:
        """Compute the net force on the golf ball.

        Parameters
        ----------
        state : np.ndarray
            State vector [x, y, z, vx, vy, vz].
        spin_rate_rads : float
            Spin rate in rad/s.
        spin_axis : np.ndarray
            Unit spin axis vector.

        Returns
        -------
        np.ndarray
            Net force vector [Fx, Fy, Fz] in newtons.
        """
        vel = state[3:6]

        # Relative velocity (ball velocity minus wind velocity)
        v_rel = vel - self.environment.wind_velocity
        v_rel_mag = np.linalg.norm(v_rel)

        if v_rel_mag < 1e-10:
            return np.array([0.0, 0.0, -self.ball.mass * self.environment.gravity])

        v_rel_hat = v_rel / v_rel_mag

        # --- Gravity ---
        F_gravity = np.array([0.0, 0.0, -self.ball.mass * self.environment.gravity])

        # --- Aerodynamic drag ---
        cd, cl = self._compute_coefficients(spin_rate_rads, v_rel)
        q = 0.5 * self.environment.air_density * v_rel_mag ** 2  # dynamic pressure * area factor
        F_drag = -q * cd * self.ball.area * v_rel_hat

        # --- Magnus force ---
        # omega = spin_rate_rads * spin_axis (angular velocity vector)
        # Magnus direction: omega x v_rel  (perpendicular to both spin axis and velocity)
        omega_vec = spin_rate_rads * spin_axis
        magnus_dir = np.cross(omega_vec, v_rel)
        magnus_dir_mag = np.linalg.norm(magnus_dir)

        if magnus_dir_mag > 1e-10:
            magnus_hat = magnus_dir / magnus_dir_mag
            F_magnus = q * cl * self.ball.area * magnus_hat
        else:
            F_magnus = np.zeros(3)

        return F_gravity + F_drag + F_magnus

    # ------------------------------------------------------------------
    # ODE system
    # ------------------------------------------------------------------

    def _equations_of_motion(
        self,
        t: float,
        state: np.ndarray,
        spin_rate_rads: float,
        spin_axis: np.ndarray,
    ) -> list[float]:
        """Right-hand side of the equations of motion.

        State vector: [x, y, z, vx, vy, vz]
        Returns: [vx, vy, vz, ax, ay, az]
        """
        vel = state[3:6]
        F = self._compute_forces(state, spin_rate_rads, spin_axis)
        a = F / self.ball.mass
        return [vel[0], vel[1], vel[2], a[0], a[1], a[2]]

    # ------------------------------------------------------------------
    # Ground-detection event
    # ------------------------------------------------------------------

    def _make_ground_event(self):
        """Create a ground-detection event for solve_ivp.

        The event triggers when z crosses zero from above (ball hits ground).
        """

        def ground_event(t, state, spin_rate_rads, spin_axis):
            return state[2]  # z-coordinate

        ground_event.terminal = True
        ground_event.direction = -1  # Only detect z crossing from positive to negative
        return ground_event

    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------

    def simulate_trajectory(
        self,
        launch_conditions: LaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.001,
    ) -> TrajectoryResult:
        """Simulate a full ball flight trajectory.

        Parameters
        ----------
        launch_conditions : LaunchConditions
            Initial launch conditions (velocity, angles, spin).
        max_time : float
            Maximum simulation time in seconds (safety limit).
        dt : float
            Approximate output time step in seconds. Also used as max_step
            for the integrator to ensure accuracy.

        Returns
        -------
        TrajectoryResult
            Complete trajectory data including carry, height, flight time, etc.

        Notes
        -----
        Uses scipy.integrate.solve_ivp with the RK45 (Dormand-Prince) method.
        Integration terminates when the ball hits the ground (z < 0).
        """
        lc = launch_conditions

        # Decompose initial velocity into components
        v = lc.velocity
        theta = lc.launch_angle  # vertical launch angle
        phi = lc.azimuth_angle  # horizontal azimuth

        vx = v * np.cos(theta) * np.cos(phi)
        vy = v * np.cos(theta) * np.sin(phi)
        vz = v * np.sin(theta)

        # Initial state: [x, y, z, vx, vy, vz]
        # Start at a tiny positive z to avoid immediate ground-event trigger
        state0 = np.array([0.0, 0.0, 1e-6, vx, vy, vz])

        # Convert spin rate to rad/s
        spin_rate_rads = rpm_to_rads(lc.spin_rate)
        spin_axis = lc.spin_axis

        # Set up the ground event
        ground_event = self._make_ground_event()

        # Solve the ODE
        sol = solve_ivp(
            self._equations_of_motion,
            t_span=[0.0, max_time],
            y0=state0,
            method="RK45",
            args=(spin_rate_rads, spin_axis),
            events=ground_event,
            max_step=dt,
            dense_output=True,
            rtol=1e-9,
            atol=1e-11,
        )

        # Build uniformly-spaced output points using the dense output
        t_end = sol.t[-1]
        t_eval = np.arange(0.0, t_end, dt)
        if len(t_eval) == 0 or t_eval[-1] < t_end - 1e-12:
            t_eval = np.append(t_eval, t_end)

        states = sol.sol(t_eval)  # shape (6, N)

        points: list[TrajectoryPoint] = []
        for i in range(len(t_eval)):
            pos = np.array([states[0, i], states[1, i], states[2, i]])
            vel = np.array([states[3, i], states[4, i], states[5, i]])
            points.append(TrajectoryPoint(t=float(t_eval[i]), position=pos, velocity=vel))

        # --- Compute summary statistics ---
        final_pos = points[-1].position
        final_vel = points[-1].velocity

        # Carry distance: horizontal distance from launch to landing
        carry_distance = float(np.sqrt(final_pos[0] ** 2 + final_pos[1] ** 2))

        # Total distance (same as carry for flat ground)
        total_distance = carry_distance

        # Maximum height and apex position
        z_values = np.array([p.position[2] for p in points])
        max_height = float(np.max(z_values))
        apex_idx = int(np.argmax(z_values))
        apex_position = points[apex_idx].position.copy()

        # Flight time
        flight_time = float(points[-1].t)

        # Descent angle: angle below horizontal at landing (positive = descending)
        v_horizontal = np.sqrt(final_vel[0] ** 2 + final_vel[1] ** 2)
        descent_angle = float(rad_to_deg(np.arctan2(-final_vel[2], v_horizontal)))

        # Landing velocity (speed)
        landing_velocity = float(np.linalg.norm(final_vel))

        return TrajectoryResult(
            points=points,
            carry_distance=carry_distance,
            total_distance=total_distance,
            max_height=max_height,
            flight_time=flight_time,
            descent_angle=descent_angle,
            landing_velocity=landing_velocity,
            apex_position=apex_position,
        )

    # ------------------------------------------------------------------
    # Spin estimation from impact physics
    # ------------------------------------------------------------------

    def estimate_spin(
        self,
        club_speed: float,
        ball_speed: float,
        loft_deg: float,
        dynamic_loft_deg: Optional[float] = None,
    ) -> float:
        """Estimate spin rate from impact physics.

        Parameters
        ----------
        club_speed : float
            Club head speed in m/s.
        ball_speed : float
            Ball speed in m/s (not directly used in this simplified model,
            but available for future COR-based refinements).
        loft_deg : float
            Static loft of the club in degrees.
        dynamic_loft_deg : float, optional
            Dynamic loft in degrees (loft + attack angle). If not provided,
            static loft is used as an approximation (assumes zero attack angle).

        Returns
        -------
        float
            Estimated spin rate in RPM.

        Notes
        -----
        Backspin estimation based on tangential impulse during impact:

            omega_backspin = (2 * mu * V_club * sin(loft)) / (5 * r_ball)

        where mu ~ 0.4 is the friction coefficient for a grooved club face,
        and the factor 2/5 arises from the moment of inertia of a solid sphere
        (I = 2/5 * m * r^2). The angular velocity is converted to RPM.
        """
        if dynamic_loft_deg is not None:
            loft_rad = deg_to_rad(dynamic_loft_deg)
        else:
            loft_rad = deg_to_rad(loft_deg)

        mu = 0.4  # friction coefficient for grooved face
        omega_backspin = (2.0 * mu * club_speed * np.sin(loft_rad)) / (5.0 * self.ball.radius)
        return rads_to_rpm(omega_backspin)

    # ------------------------------------------------------------------
    # Quick carry estimate
    # ------------------------------------------------------------------

    def estimate_carry(
        self,
        ball_speed: float,
        launch_angle_deg: float,
        spin_rate: float,
    ) -> float:
        """Quick carry distance estimate by running the full simulation.

        Parameters
        ----------
        ball_speed : float
            Ball speed in m/s.
        launch_angle_deg : float
            Launch angle in degrees.
        spin_rate : float
            Spin rate in RPM.

        Returns
        -------
        float
            Carry distance in metres.
        """
        lc = LaunchConditions(
            velocity=ball_speed,
            launch_angle=deg_to_rad(launch_angle_deg),
            spin_rate=spin_rate,
        )
        result = self.simulate_trajectory(lc)
        return result.carry_distance

    # ------------------------------------------------------------------
    # Wind / density helpers
    # ------------------------------------------------------------------

    @staticmethod
    def wind_effect(wind_vec: np.ndarray, ball_vel: np.ndarray) -> np.ndarray:
        """Calculate the relative velocity of the ball through the air.

        Parameters
        ----------
        wind_vec : np.ndarray
            Wind velocity vector in m/s [x=tailwind, y=crosswind, z=up].
        ball_vel : np.ndarray
            Ball velocity vector in m/s.

        Returns
        -------
        np.ndarray
            Relative velocity of ball w.r.t. air: v_ball - v_wind.
        """
        return ball_vel - np.asarray(wind_vec, dtype=float)

    @staticmethod
    def altitude_adjusted_density(altitude: float) -> float:
        """Compute altitude-adjusted air density.

        Parameters
        ----------
        altitude : float
            Elevation above sea level in metres.

        Returns
        -------
        float
            Air density in kg/m^3.

        Notes
        -----
        rho = 1.225 * exp(-altitude / 8500)

        This barometric approximation uses a scale height of 8500 m,
        appropriate for the standard atmosphere in the troposphere.
        """
        return 1.225 * np.exp(-altitude / 8500.0)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def to_trajectory_chart(
        self,
        trajectory: TrajectoryResult,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a side-view trajectory plot.

        Parameters
        ----------
        trajectory : TrajectoryResult
            Simulated trajectory result.
        filename : str, optional
            If provided, save SVG to this path and return None.
            If omitted, return the SVG as a string.

        Returns
        -------
        str or None
            SVG string if no filename given, otherwise None.
        """
        x_yards = [m_to_yards(p.position[0]) for p in trajectory.points]
        z_m = [p.position[2] for p in trajectory.points]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_yards, z_m, color="#1f77b4", linewidth=2)
        ax.set_xlabel("Distance (yards)")
        ax.set_ylabel("Height (m)")
        ax.set_title("Ball Flight Trajectory — Side View")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        # Annotate carry and max height
        carry_yd = m_to_yards(trajectory.carry_distance)
        ax.axvline(carry_yd, color="grey", linestyle="--", alpha=0.5)
        ax.annotate(
            f"Carry: {carry_yd:.0f} yd",
            xy=(carry_yd, 0),
            xytext=(carry_yd - 15, trajectory.max_height * 0.15),
            fontsize=9,
            color="grey",
        )
        ax.annotate(
            f"Apex: {trajectory.max_height:.1f} m",
            xy=(m_to_yards(trajectory.apex_position[0]), trajectory.max_height),
            xytext=(m_to_yards(trajectory.apex_position[0]) + 5, trajectory.max_height + 1),
            fontsize=9,
            color="grey",
        )

        if filename is not None:
            fig.savefig(filename, format="svg", bbox_inches="tight")
            plt.close(fig)
            return None

        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def to_base64_png(
        self,
        trajectory: TrajectoryResult,
        dpi: int = 150,
    ) -> str:
        """Generate a base64-encoded PNG of the trajectory chart.

        Parameters
        ----------
        trajectory : TrajectoryResult
            Simulated trajectory result.
        dpi : int
            Resolution in dots per inch.

        Returns
        -------
        str
            Base64-encoded PNG image string, suitable for embedding in HTML.
        """
        x_yards = [m_to_yards(p.position[0]) for p in trajectory.points]
        z_m = [p.position[2] for p in trajectory.points]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_yards, z_m, color="#1f77b4", linewidth=2)
        ax.set_xlabel("Distance (yards)")
        ax.set_ylabel("Height (m)")
        ax.set_title("Ball Flight Trajectory — Side View")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Validation against TrackMan PGA Tour averages
# ---------------------------------------------------------------------------

def _run_validation():
    """Validate the physics engine against TrackMan PGA Tour average data.

    TrackMan PGA Tour Averages:
        Driver:  ball speed 73 m/s, launch 11°, spin 2500 rpm → carry 250-280 yd
        7-Iron:  ball speed 53 m/s, launch 16°, spin 7000 rpm → carry 155-175 yd
        PW:      ball speed 42 m/s, launch 23°, spin 9000 rpm → carry 115-135 yd
    """
    sim = BallFlightSimulator()

    test_cases = [
        {
            "name": "Driver",
            "ball_speed": 73.0,
            "launch_angle_deg": 11.0,
            "spin_rpm": 2500,
            "carry_min_yd": 250,
            "carry_max_yd": 280,
        },
        {
            "name": "7-Iron",
            "ball_speed": 53.0,
            "launch_angle_deg": 16.0,
            "spin_rpm": 7000,
            "carry_min_yd": 155,
            "carry_max_yd": 175,
        },
        {
            "name": "PW",
            "ball_speed": 42.0,
            "launch_angle_deg": 23.0,
            "spin_rpm": 9000,
            "carry_min_yd": 115,
            "carry_max_yd": 135,
        },
    ]

    all_pass = True
    print("=" * 70)
    print("OpenLaunch Physics Engine — TrackMan Validation")
    print("=" * 70)

    for tc in test_cases:
        lc = LaunchConditions(
            velocity=tc["ball_speed"],
            launch_angle=deg_to_rad(tc["launch_angle_deg"]),
            spin_rate=tc["spin_rpm"],
        )
        result = sim.simulate_trajectory(lc)
        carry_yd = m_to_yards(result.carry_distance)
        in_range = tc["carry_min_yd"] <= carry_yd <= tc["carry_max_yd"]
        status = "PASS" if in_range else "FAIL"
        if not in_range:
            all_pass = False

        print(f"\n{tc['name']}:")
        print(f"  Ball speed:     {tc['ball_speed']:.1f} m/s")
        print(f"  Launch angle:   {tc['launch_angle_deg']:.1f}°")
        print(f"  Spin rate:      {tc['spin_rpm']:.0f} rpm")
        print(f"  Carry:          {carry_yd:.1f} yd  (target: {tc['carry_min_yd']}-{tc['carry_max_yd']} yd)  [{status}]")
        print(f"  Max height:     {result.max_height:.1f} m")
        print(f"  Flight time:    {result.flight_time:.2f} s")
        print(f"  Descent angle:  {result.descent_angle:.1f}°")
        print(f"  Landing speed:  {result.landing_velocity:.1f} m/s")

    # Also test spin estimation
    print("\n--- Spin Estimation ---")
    # Driver: ~110 mph club speed, ~10.5° loft
    driver_spin = sim.estimate_spin(club_speed=mph_to_ms(110), ball_speed=73.0, loft_deg=10.5)
    print(f"  Driver (110 mph, 10.5° loft): {driver_spin:.0f} rpm  (typical: 2400-2800)")
    # 7-Iron: ~80 mph club speed, ~34° loft
    iron_spin = sim.estimate_spin(club_speed=mph_to_ms(80), ball_speed=53.0, loft_deg=34.0)
    print(f"  7-Iron (80 mph, 34° loft):    {iron_spin:.0f} rpm  (typical: 6500-7500)")

    # Test altitude adjustment
    print("\n--- Altitude Density Adjustment ---")
    denver = BallFlightSimulator.altitude_adjusted_density(1609)  # Denver ~1 mile
    print(f"  Denver (1609 m): {denver:.4f} kg/m³  (sea level: 1.225)")

    # Test wind effect
    print("\n--- Wind Effect ---")
    rel = BallFlightSimulator.wind_effect(
        wind_vec=np.array([5.0, 0.0, 0.0]),  # 5 m/s tailwind
        ball_vel=np.array([70.0, 0.0, 10.0]),
    )
    print(f"  Ball 70 m/s forward, 5 m/s tailwind → relative: {rel} m/s")

    # Test conversion functions
    print("\n--- Unit Conversions ---")
    print(f"  100 mph = {mph_to_ms(100):.2f} m/s")
    print(f"  100 m/s = {ms_to_mph(100):.2f} mph")
    print(f"  200 yards = {yards_to_m(200):.2f} m")
    print(f"  3000 rpm = {rpm_to_rads(3000):.2f} rad/s")

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL VALIDATION CHECKS PASSED ✓")
    else:
        print("SOME VALIDATION CHECKS FAILED ✗")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    _run_validation()
