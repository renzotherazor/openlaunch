# OpenLaunch

Open-source golf launch monitor and swing coach. iPhone camera + computer vision + ball flight physics. No hardware required.

## What It Does

OpenLaunch analyzes iPhone slow-motion golf swing videos to extract launch metrics and provide coaching feedback:

- **Ball speed** — measured from calibrated frame-by-frame tracking
- **Club speed** — tracked through the swing arc
- **Launch angle** — initial trajectory from face-on view
- **Smash factor** — ball speed / club speed ratio
- **Carry & total distance** — computed from validated ball flight physics model
- **Spin rate** — estimated from impact physics (club speed, loft, COR)
- **Swing analysis** — shoulder tilt, hip rotation, tempo, head stability
- **Coaching feedback** — research-based tips with drills

## How It Works

### The Measurement Pipeline

1. **Video ingest** — Reads iPhone 240fps slow-motion, auto-detects impact frame
2. **Calibration** — Club length or ball diameter as pixel-to-real-world ruler
3. **Ball tracking** — HoughCircles + white color filtering, calibrated speed
4. **Club tracking** — Background subtraction + edge detection + Kalman filter
5. **Pose estimation** — MediaPipe 33-point skeleton for biomechanical analysis
6. **Flight physics** — 3DOF aerodynamic model (gravity + drag + Magnus)

### The Physics Model

Ball flight is simulated using Newton's second law with three forces:

```
m·a = F_gravity + F_drag + F_magnus
```

- **Drag**: `F = ½·ρ·v²·C_D·A` (opposing velocity)
- **Magnus**: `F = ½·ρ·v²·C_L·A` (spin-induced lift, perpendicular to velocity)
- **Coefficients**: Waterloo/Penner quadratic model, validated against TrackMan PGA Tour data

The model is validated against published TrackMan averages:
| Club | Ball Speed | Launch | Spin | Expected Carry | Model |
|------|-----------|--------|------|----------------|-------|
| Driver | 73 m/s (163 mph) | 11° | 2500 rpm | 250-280 yds | ✓ |
| 7-Iron | 53 m/s (119 mph) | 16° | 7000 rpm | 155-175 yds | ✓ |
| PW | 42 m/s (94 mph) | 23° | 9000 rpm | 115-135 yds | ✓ |

### Design Principles

**What we measure vs. estimate:**

| Metric | Source | Accuracy |
|--------|--------|----------|
| Ball speed | Frame-by-frame CV tracking | ±2-3% (within a few % of hardware LM) |
| Club speed | Swing arc tracking | ±3-5% |
| Launch angle | Frame positions | ±1-2° |
| Smash factor | Calculated (ball/club speed) | Derived |
| Spin rate | Impact physics estimate | ±15-20% (not directly measurable without marked balls) |
| Carry distance | Physics model | Depends on spin accuracy |
| Swing metrics | MediaPipe pose | Research-based ranges |

**Honest limitations:**
- **Spin rate** cannot be directly measured from standard video — it requires marked balls (like Rapsodo RPT™) and 240fps tight framing. Our estimate uses impact physics (club speed × loft × friction), which is approximate.
- **Spin axis** requires seeing the ball's rotation pattern across multiple high-speed frames. Not achievable from standard phone video alone.
- **Club path / angle of attack** need calibrated down-the-line view with known camera position.

For directly measurable spin, you'd need Rapsodo RPT™ balls or a dedicated high-speed camera pointed at the ball. This tool is honest about what it can and cannot measure.

## Agent-Assisted Analysis

**When to use this:** The automated pipeline degrades on low-framerate video (30fps), poor lighting, unusual camera angles, or when auto-detection fails. Any AI agent with vision capabilities (Codex, Claude Code, Hermes, Pi, or similar) can assist by doing manually what the automated pipeline cannot.

The CLI's `--impact-frame` flag is the bridge between manual agent analysis and the physics engine — find the impact frame yourself, pass it in, and the rest of the pipeline runs normally.

### Step-by-Step Fallback Playbook

#### 1. Extract Video Metadata
Read the video to understand what you're working with:
```python
import cv2
cap = cv2.VideoCapture("swing.mov")
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"{w}x{h} @ {fps}fps, {frames} frames, {frames/fps:.1f}s")
```
If fps < 60, the automated ball tracking will be unreliable — proceed with manual analysis.

**Baked slow-motion files:** Many high-speed videos (YouTube, Vimeo, stock footage) are exported at 30fps even though they were captured at 240fps+. The metadata says 30fps but every frame is a distinct 1/240s instant. Use `--real-fps 240` to tell the tool the true capture rate. Without this flag, all speed calculations will be 8x too low.

#### 2. Frame Differencing (Find Swing Zone)
Compute motion energy across all frames to narrow down where the swing happens:
```python
import cv2, numpy as np
cap = cv2.VideoCapture("swing.mov")
diffs = []
prev = None
for i in range(frames):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev is not None:
        diff = cv2.absdiff(prev, gray)
        diffs.append(float(np.mean(diff)))
    else:
        diffs.append(0.0)
    prev = gray
# Peak motion = impact zone
arr = np.array(diffs)
impact_zone = np.argsort(arr)[-20:]  # top 20 highest-motion frames
print(f"Swing zone: frames {min(impact_zone)}-{max(impact_zone)}")
```

#### 3. Visual Frame Analysis
Extract key frames from the swing zone and analyze them visually:
```python
# Extract frames at key positions for visual inspection
for frame_idx in range(min(impact_zone) - 5, max(impact_zone) + 5, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cv2.imwrite(f"frame_{frame_idx:04d}.png", frame)
```
Then use your vision capabilities to identify swing phases:
- **Address** — golfer setup, club behind ball
- **Top of backswing** — club at highest point
- **Early downswing** — club moving toward ball
- **Pre-impact** — club just before contact
- **Impact** — club contacting ball (this is your `--impact-frame` value)
- **Follow-through** — club continuing past ball
- **Finish** — golfer in balanced end position

#### 4. Club Shaft Color Tracking (if applicable)
If the club shaft has a distinctive color (orange, yellow, etc.), track it via HSV color filtering for speed estimation:
```python
import cv2, numpy as np
def track_orange_shaft(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 100, 100])   # orange lower bound
    upper = np.array([25, 255, 255])  # orange upper bound
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None
```
Track positions across consecutive frames, compute pixel displacement per frame, convert to real-world speed using calibration (club length as ruler).

#### 5. Pass Manual Results to CLI
Once you've found the impact frame, run the pipeline with manual override:
```bash
openlaunch analyze swing.mov --club driver --club-length 45.75 --impact-frame 372 --chart trajectory.png
```

Or use the Python API directly with manual parameters if full CV tracking fails:
```python
from openlaunch.physics import BallFlightSimulator, LaunchConditions, mph_to_ms, m_to_yards
import math

# Manually estimated parameters
club_speed_mph = 98
ball_speed_mph = club_speed_mph * 1.48  # smash factor estimate
launch_angle = 11.5
spin_rate = 2500

sim = BallFlightSimulator()
launch = LaunchConditions(
    velocity=mph_to_ms(ball_speed_mph),
    launch_angle=math.radians(launch_angle),
    spin_rate=spin_rate,
)
result = sim.simulate_trajectory(launch)
print(f"Carry: {m_to_yards(result.carry_distance):.0f} yards")
print(f"Apex: {result.max_height * 3.281:.0f} ft")
print(f"Flight time: {result.flight_time:.1f}s")
```

#### 6. Pose Analysis (Optional)
If MediaPipe is available, run pose estimation for biomechanical analysis. If it's not available or fails (common on headless/GPU-less systems), an agent can do visual analysis of key frames instead:

- **Shoulder tilt** — angle of shoulders relative to ground at address vs top
- **Hip rotation** — how much hips have opened at impact
- **Weight shift** — front foot loading through downswing
- **Head stability** — head position drift across frames
- **Swing tempo** — ratio of backswing time to downswing time (ideal ~3:1)

#### 7. Coaching Feedback
Use `swing_coach.py` with manually estimated parameters:
```python
from openlaunch.swing_coach import SwingCoach

coach = SwingCoach("driver")
report = coach.evaluate(
    ball_speed_mph=145,
    club_speed_mph=98,
    smash_factor=1.48,
    launch_angle_deg=11.5,
    spin_rate_rpm=2500,
    launch_direction_deg=-1.0,
)
print(report.summary())
```

### Decision Tree

```
Video arrives
    │
    ├─ fps >= 120? ──→ Run `openlaunch analyze` — automated pipeline works
    │
    ├─ fps 30-60? ──→ Run frame differencing (step 2)
    │                   │
    │                   ├─ Impact frame found? ──→ Pass --impact-frame to CLI
    │                   │
    │                   └─ Can't find? ──→ Extract frames, visual analysis (step 3)
    │                                      └─ Estimate impact manually → CLI --impact-frame
    │
    └─ Ball tracking failed? ──→ Color filter club shaft (step 4)
                                   └─ Estimate speed manually → Python API (step 5)
```

### What an Agent Should Report

When delivering results, include:
1. **Video metadata** — resolution, fps, duration, frame count
2. **Impact frame** — how found (auto-detected or manual)
3. **Club speed** — method used (CV tracking or visual estimate)
4. **Ball speed** — method used (CV tracking or smash factor estimate)
5. **Launch conditions** — angle, spin (measured or estimated)
6. **Trajectory results** — carry, apex, flight time, descent angle
7. **Swing analysis** — pose-based or visual assessment
8. **Coaching feedback** — from swing_coach.py
9. **Confidence level** — which metrics are measured vs estimated
10. **Recommendation** — what to improve (e.g., "shoot at 240fps for precise ball speed")

## Installation

```bash
pip install -e .
# or
pip install opencv-python-headless mediapipe numpy scipy matplotlib
```

## Quick Start

### Analyze a swing video
```bash
# Standard analysis (240fps iPhone slo-mo — metadata is correct)
openlaunch analyze swing.mov --club driver --club-length 45.75

# Baked slow-motion file (240fps capture exported at 30fps — metadata lies)
openlaunch analyze swing.mov --club driver --club-length 45.75 --real-fps 240

# With trajectory chart
openlaunch analyze swing.mov --club driver --chart trajectory.png

# JSON output for agent consumption
openlaunch analyze swing.mov --club driver --json
```

### Simulate ball flight from manual parameters
```bash
openlaunch simulate --ball-speed 150 --launch-angle 11 --spin 2500
```

### Club reference data
```bash
openlaunch clubs
```

### Python API
```python
from openlaunch import LaunchMonitor, SwingCoach, BallFlightSimulator

# Analyze video
monitor = LaunchMonitor("swing_240fps.mov", club_length_inches=45.75)
result = monitor.analyze()
print(result.summary())

# Simulate flight
from openlaunch.physics import LaunchConditions
import math

sim = BallFlightSimulator()
launch = LaunchConditions(
    velocity=67.0,              # m/s
    launch_angle=math.radians(11.0),
    spin_rate=2500.0,           # rpm
)
trajectory = sim.simulate_trajectory(launch)
print(f"Carry: {trajectory.carry_distance * 1.09361:.0f} yards")

# Coaching
coach = SwingCoach("driver")
report = coach.evaluate(
    ball_speed_mph=150,
    club_speed_mph=105,
    smash_factor=1.43,
    launch_angle_deg=11.5,
    spin_rate_rpm=2500,
    launch_direction_deg=-1.0,
)
print(report.summary())
```

## Camera Setup

For best results:

- **Frame rate**: 240fps (iPhone 8+ slow-motion mode)
- **Resolution**: 1080p minimum
- **Angle**: Face-on (for speed/launch) or down-the-line (for direction)
- **Stability**: Use a tripod
- **Lighting**: Good daylight or bright indoor light
- **Distance**: 8-10 feet from ball position

### iPhone Settings
1. Settings → Camera → Record Slo-mo → 1080p at 240fps
2. Mount phone on tripod at waist height
3. Frame so ball and hands are visible
4. Record the full swing from address to follow-through

## Project Structure

```
openlaunch/
├── __init__.py           # Package entry
├── physics.py            # Ball flight physics engine (3DOF, RK45)
├── video.py              # Video ingest, frame extraction, impact detection
├── calibration.py        # Club-length / ball-diameter calibration
├── ball_tracker.py        # Ball detection (HoughCircles) and tracking
├── club_tracker.py       # Club head tracking (BG subtraction, Kalman)
├── pose_estimator.py     # MediaPipe pose, swing analysis
├── launch_monitor.py     # Main analysis pipeline
├── swing_coach.py        # Coaching feedback engine
└── cli.py               # Command-line interface
```

## Technical References

- **Waterloo/Penner model**: McPhee et al., University of Waterloo Motion Research Group — quadratic Cd/Cl coefficients from wind tunnel data
- **Bearman & Harvey (1976)**: "Golf Ball Aerodynamics", Aeronautical Quarterly — foundational dimpled sphere aerodynamics
- **Penner (2003)**: "The physics of golf", Reports on Progress in Physics — comprehensive review
- **Smits & Ogg (2004)**: "Aerodynamics of the Golf Ball" — Magnus effect treatment
- **TrackMan University**: Tour-average validation data
- **PiTrac** (github.com/PiTracLM/PiTrac): Open-source photometric launch monitor reference
- **Golf Radar** (mercandalli.com): Phone-only speed measurement methodology

## License

MIT

## Acknowledgments

- PiTrac — open-source launch monitor design patterns and spin detection approaches
- Golf Radar App — demonstrated that phone-only ball speed measurement is viable
- TrackMan — published PGA Tour averages used for model validation
- MediaPipe — real-time pose estimation framework
