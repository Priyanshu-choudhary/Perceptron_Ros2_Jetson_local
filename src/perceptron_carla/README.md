# perceptron_carla

CARLA simulator bridge for the Perceptron rover. CARLA runs natively on Windows; this package
runs in WSL (ROS 2 Humble) and turns the simulation into the same topics, frames and message
shapes the real rover produces, plus ground truth for scoring. Roadmap stages 0-3 live here.

```
Windows: CarlaUE4.exe (0.9.15) ──TCP 2000-2002 (+8000 Traffic Manager)──► WSL: carla_bridge
                                                                         ├─ robot-equivalent topics ──► EKFs, perception
                                                                         └─ ground truth ───────────► evaluators only
```

## Stage 0: baseline

| Item | Value |
|---|---|
| Simulator | CARLA 0.9.15 (UE4), Windows package, Town10HD_Opt, synchronous mode at 20 Hz |
| Client | `pip install --user carla==0.9.15` in WSL (cp310 wheel). Must match the server exactly |
| ROS | ROS 2 Humble, Ubuntu 22.04.5 (WSL distro `ros2-humble`), robot_localization 3.5.4 |
| Network | WSL2 NAT. The bridge finds the Windows host as the default gateway; set `CARLA_HOST` to override |
| Bridge revision | this package in git, branch `feature/carla-sim-perception` |
| Baseline bag | `~/bags/baseline_<date>`: 20 s, autopilot + 15 cars + 20 pedestrians, all robot-equivalent topics + ground-truth boxes (965 MiB) |
| Example frames | `Carl_sim/samples/` on Windows (RGB, semantic, instance, boxes, lidar overlay, trajectories) |

**Timing.** The bridge ticks the world, then waits for every sensor's data **for that tick**
(`frame` match) before publishing, so every topic in one tick carries the same sim-time stamp.
GNSS ticks at `--gnss-hz` and keeps the stamp of the tick it was sampled on. `/clock` is
published every tick; run every node with `use_sim_time:=true`.

**QoS.** Images, point cloud and `/imu/data` are BEST_EFFORT/KEEP_LAST 5, like the real
Pixhack driver; `/odom`, `/gps/fix`, ground truth are RELIABLE; `/tf_static` and
`/carla/camera/semantic_labels` are TRANSIENT_LOCAL. In RViz set the Image/PointCloud2
displays to Best Effort.

**Throughput** (wall clock, RTX 3080 Ti): about 13 Hz with every output on, about 18-20 Hz with
`--no-semantic --no-instance --no-boxes --no-depth`. Sim time is always exactly 20 Hz.

## Stage 1: rover geometry and frames

```
carla_world ─► map ─► odom ─► base_footprint ─► base_link ─┬─► camera_link ─► camera_optical_link
                                                          ├─► lidar3d_link
                                                          ├─► imu_link
                                                          └─► gps_link
```

**TF ownership: exactly one publisher per transform** (checked by `sensor_check`):

| Transform | `--tf-mode gt` (default) | `--tf-mode ekf` |
|---|---|---|
| carla_world → map | bridge, static identity | bridge, static: the start pose (see below) |
| map → odom | bridge, static identity | `ekf_filter_node_global`, or a static identity from `sim_localization.launch.py global:=false` |
| odom → base_footprint | bridge, ground truth, every tick | `ekf_filter_node_local` |
| base_footprint → … sensors | bridge, static | bridge, static |

`carla_world` is CARLA's world with y flipped, i.e. **ENU** (x east, y north, z up). The EKFs
start at the origin with the IMU's absolute (ENU) heading, so in ekf mode their `map`/`odom`
frames are ENU anchored where the robot stood when they started. That's why the bridge holds the
car still on its brakes for `--hold-seconds` (default 10 s in ekf mode) and anchors
`carla_world → map` at that pose.

**Sensor extrinsics** (ROS axes, parent `base_link`; `--mount rover`, the default):

| Frame | x | y | z | roll / pitch / yaw (rad) | Source |
|---|---|---|---|---|---|
| base_link (from base_footprint) | 0 | 0 | 0.0625 | 0 | URDF `base_footprint_joint` (wheel radius) |
| camera_link | rig | 0 | 0.3575 | 0 / −0.07027 / 0 | URDF `camera_joint`: lens 0.42 m above the floor, ArUco-solved pitch |
| camera_optical_link (from camera_link) | 0 | 0 | 0 | −π/2 / 0 / −π/2 | URDF `camera_optical_joint` |
| lidar3d_link | rig | 0 | 0.3525 | 0 / 0.26 / 0 | URDF `lidar3d_joint`: 16-beam, 15° down, ±15° FOV, 300k pts/s |
| imu_link | −0.0252 | 0.0066 | −0.0089 | 0 | URDF `imu_joint` |
| gps_link | 0 | 0 | 0.40 | 0 | **assumed**: not in the URDF |

**The one deliberate deviation from the URDF is `rig`.** The rover's sensors sit on its lidar
axis at x = 0, but in CARLA that point is inside the host vehicle's body. So the camera and
lidar ride on a rig **`front_bumper − rear_axle + 0.10 m`** ahead of `base_footprint` (2.04 m
on the Microlino), at the rover's true heights and pitches. Perception sees what the rover would
see; `base_footprint` stays on the rear axle, the only point of a car-like vehicle with no
sideways velocity, which the wheel-odometry model and the EKF's vy = 0 assume.
`--mount car` restores the old roof layout (camera 1.6 m, 64-beam lidar 1.9 m).

**Host vehicle and control interface** (`--vehicle`, default `vehicle.micro.microlino`, the
smallest 4-wheel CARLA vehicle: 2.2 × 1.5 × 1.4 m, wheelbase 1.525 m, max steer 70°):

* Autopilot: CARLA Traffic Manager (`--autopilot`), `--ego-speed-diff` % under the speed limit
  (default 30 %). This is how Stages 0-9 drive the car.
* Direct control (for the roadmap's Stage 10 adapter): `vehicle.apply_control(carla.VehicleControl(
  throttle∈[0,1], steer∈[−1,1] (×70°), brake∈[0,1], hand_brake, reverse))`, or
  `vehicle.apply_ackermann_control(carla.VehicleAckermannControl(steer [rad], steer_speed,
  speed [m/s], acceleration, jerk))`. Kinematics are Ackermann, not the rover's skid steer.

## Stage 2: simulated robot sensors

Same topic names, frames, QoS and covariances as the real drivers, so `dual_ekf_navsat.yaml`
runs unchanged. Noise models: `perceptron_carla/sim_sensors.py`; every value is a bridge option.

| Topic | Stands in for | Content and noise (defaults) |
|---|---|---|
| `/odom` (odom → base_footprint) | `stm32_bridge_node` | twist only (vx, wz) at every tick; pose at identity with 1e6 covariance. +2 % speed and +5 % yaw-rate scale error, 0.02 m/s and 0.01 rad/s white noise. Never the true pose |
| `/imu/data` (imu_link) | `pixhack_bridge_node` | **REP-103**: ENU orientation (yaw 0 = east, CCW+). Heading: +2° constant offset + 0.05°/√s random walk + 0.5° noise; roll/pitch 0.3° noise. Gyro from CARLA's IMU with 0.005 rad/s noise, 0.002 rad/s bias. Acceleration includes gravity, like the Pixhack |
| `/gps/fix` (gps_link) | `pixhack_bridge_node` | 10 Hz, 1.0 m white noise + 0.5 m per-run offset. Re-anchored at `--gnss-origin` (default 19.1334 N, 72.9133 E): CARLA's own geo-reference is lat 0 / lon 0, on the equator and the UTM zone 30/31 boundary, where navsat's UTM conversion jumps |
| `/carla/ground_truth/ego_odom` | evaluation only | perfect base_footprint pose and twist in carla_world. **Never** an EKF input |

Conventions measured on 0.9.15, not assumed (pinned by `test/test_sim_sensors.py`): CARLA +x
is east and +y is south; gyro z follows CARLA's clockwise yaw (negated for ROS); the
accelerometer reads +9.81 on z at rest.

Check it any time (prints rates, errors vs truth and the TF tree; exit code 1 on failure):

```bash
ros2 run perceptron_carla sensor_check --duration 30
```

Measured (rover mount, autopilot, 15 cars, 15 pedestrians): speed scale 1.020, residual
0.020 m/s; yaw-rate residual 0.011 rad/s; IMU heading +1.98° ± 0.52°, gyro z 0.997 correlated
with truth, gravity 9.81; GNSS displacement error median 1.4 m; TF tree PASS.

## Stage 3: EKF localization validation

```bash
# terminal 1 (start first, it waits for /clock)
ros2 launch perceptron_carla sim_localization.launch.py                   # your setup, exactly
ros2 launch perceptron_carla sim_localization.launch.py fixes:=true       # with the fixes below
# terminal 2
ros2 run perceptron_carla carla_bridge --autopilot --tf-mode ekf --no-semantic --no-instance --no-boxes --no-depth
#   add --odom-vy-cov 0.001 together with fixes:=true
```

The launch file runs the real `dual_ekf_navsat.yaml` (plus `sim_ekf_overrides.yaml`: sim time,
and `yaw_offset: 0` because the sim IMU is ENU) with the same node names and remaps as
`perceptron_edge` GPS mode. `localization_eval_node` scores `/odometry/local` (dead reckoning
from the start) and `/odometry/global` against ground truth, interpolated to each estimate's
stamp, and writes `~/carla_logs/loc_eval_<time>.csv`.

### Stage 3 findings: four problems in the current localization setup

Found by running your stack on simulated sensors. All four also apply to the real rover. None
of the real robot's files were changed; each fix is shown here and switchable in the sim.

1. **vy is unmeasured.** `stm32_bridge_node` publishes `/odom` vy with variance **1e6**, so
   nothing constrains sideways velocity and the local EKF lets it wander (mean 1.7 m/s, peaks of
   9 m/s at 3 m/s driving). *Fix:* publish vy = 0 with a small variance (sim:
   `--odom-vy-cov 0.001`; real: `TWIST_COV_DIAG[1]` in `stm32_bridge_node.py`).
2. **The local EKF fuses a fake pose.** `odom0_config` fuses x and y from `/odom`, which the
   driver always publishes as (0, 0). As the EKF's own uncertainty grows, that "you are at the
   origin" measurement gains weight and drags the estimate back. *Fix:* set odom0 x/y to false
   in `ekf_filter_node_local`.
3. **navsat never receives the IMU.** In ROS 2, `navsat_transform_node`'s IMU topic is `imu`
   (it was `imu/data` in ROS 1). `edge.launch.py`'s remap `('/imu/data', '/imu/data')` is a
   no-op, so navsat never gets a heading, never finishes initialising, and re-sets its datum on
   **every** GPS fix (thousands of "Datum" log lines). `/odometry/gps` then just echoes the
   current pose, so GPS adds nothing. *Fix:* remap `('imu', '/imu/data')`.
4. **navsat is fed the local EKF.** With fix 3 alone, navsat's GPS odometry comes out in `odom`
   and the global EKF moves it into `map` with its own `map → odom`: a feedback loop that ran
   to 944 m of error in the test. robot_localization's dual-EKF example feeds navsat
   `odometry/global`. *Fix:* remap navsat's `odometry/filtered` to `/odometry/global`.

Results on the same kind of autopilot route (seed 7, about 0.9-1.3 km each, rover mount):

| Run | Setup | Local RMSE / drift | Global RMSE (max) | Heading RMSE |
|---|---|---|---|---|
| A | your setup, exactly | 130 m / 16.8 % | **266 m** (448 m) | 2.7° |
| B | + fix 1 | 86 m / 11.0 % | 6.6 m (12.5 m)¹ | 2.7° |
| D | + fixes 1, 2 | 6.5 m / 0.2 % | 6.5 m (12.5 m)¹ | 2.7° |
| E | + fixes 1, 2, 3 | 6.6 m / 1.0 % | 481 m (944 m) | 2.7° |
| **F** | **+ fixes 1, 2, 3, 4** | **6.5 m / 0.74 %** | **1.34 m (2.38 m)** | 2.7° |

¹ Without fix 3 the global EKF only looks reasonable because navsat re-datums onto the current
pose every fix. It's the local estimate with extra steps, not a GPS correction.

**What limits it now** is the IMU heading offset (2° by default). The EKF trusts the IMU's
absolute yaw, so heading error ≈ the offset; with `--yaw-bias-deg 0` the GPS-corrected error
halved in a separate run. On the rover, compass calibration (and `magnetic_declination_radians`)
will matter most. The EKF update rate in sim is 20 Hz: timers run on `/clock`, which advances
once per 20 Hz tick, so the configured 30 Hz can't be reached.

**Real-robot note (not tested in sim):** `pixhack_bridge_node` publishes MAVLink ATTITUDE angles
directly, and those are NED/FRD (yaw 0 = north, clockwise positive), while ROS expects ENU
(counter-clockwise). `yaw_offset: π/2` fixes the zero but not the direction. Check the sign of
`/imu/data` yaw rate against wheel yaw rate on the robot.

**Guard.** `localization_eval_node` reads `/carla/bridge_info` and refuses to score (with a
message saying why) unless the bridge runs with `--tf-mode ekf`. Otherwise it would score the
distance from the world origin while the bridge and the EKF fight over `odom → base_footprint`.

## LiDAR model and scenarios (added in Stage 4)

* **Dropout and noise.** CARLA's default `dropoff_general_rate` 0.45 silently discards 45 % of
  all returns. The bridge sets it to 0 (`--lidar-dropoff`) and adds 1.5 cm range noise
  (`--lidar-noise`), like a VLP-16. About 14.5 k points per sweep instead of about 8 k.
* **Semantic twin.** `/carla/ground_truth/lidar_semantic` has the same rays from the same pose, each with
  `obj_idx` (actor id) and `obj_tag` (class). Evaluation only (`--no-lidar-gt` to turn off).
* **`/carla/bridge_info`** (latched JSON): `tf_mode`, mount, vehicle, `ego_id`, rig position and
  the host body box in `base_footprint`. The box is also logged at start-up; the preprocessor's
  self box is that plus a margin.
* **`--crossing N`**: N pedestrians pacing across the road at 5, 10, 15 … m ahead of the rig at
  0.8-1.4 m/s, no AI controller, so their true positions and velocities are known at every tick.
  This is the repeatable case for clustering, tracking and prediction. The ego stays put.

**RViz preset** (Best Effort already set where needed):

```bash
rviz2 -d $(ros2 pkg prefix perceptron_carla)/share/perceptron_carla/rviz/carla.rviz
```

## RViz: lane lines and the ego car

* **`/carla/map/lanes`** (latched MarkerArray, `carla_world`): lane lines from CARLA's OpenDRIVE map, published
  once at start-up. Cyan: lane centre lines (fainter inside junctions); white / yellow: painted markings,
  solid or dashed, doubles drawn as two lines; grey: road edges with no paint. This is ground truth from the
  map; the rover has no such map, and its camera-based equivalent is `road_segmentation_node`
  (perceptron_perception). `--no-lane-markers` turns it off, and `--lane-step` sets the sampling (1.5 m).
* **`/carla/ego_marker`** (latched MarkerArray, `base_footprint`, frame-locked so it follows TF): the host car as
  primitives, built from its real bounding box and wheel positions (body, glass bubble, wheels, headlights,
  heading arrow), plus the rover sensor rig (mast, camera, LiDAR) at its true height.

Both are in the RViz preset as *Lane lines (CARLA map)* and *Ego car*, subscribed Transient Local.

## Other outputs (ground truth for perception)

Depth, semantic and instance segmentation, 2D visible/amodal and 3D boxes, box debug image,
markers, per-actor CSV: see the module docstring of `carla_bridge.py`. Detection id = CARLA actor
id everywhere. CARLA's instance ids are not actor ids, so the bridge matches them geometrically.

## Stage 9: scenario suite and regression checks

One command runs every subsystem against fixed, seeded CARLA scenes and says PASS or FAIL per metric,
so a change that makes things worse shows up as a FAIL row, not as a hunch about RViz.

```bash
# Windows: CarlaUE4.exe running.  WSL, nothing else running:
ros2 run perceptron_carla scenario_suite                    # all 7 scenarios, ~18 min
ros2 run perceptron_carla scenario_suite crossing city      # some of them
ros2 run perceptron_carla scenario_suite --list             # what each one is
ros2 run perceptron_carla scenario_suite --bag full         # bags with lidar + camera too (large)
```

For each scenario the runner (`scenario_suite.py`):
1. starts `carla_bridge` with the scenario's seeded arguments, writing the ground-truth CSV into the run directory;
2. starts the stack and its evaluators (`launch/scenario_stack.launch.py`), and `ros2 bag record`;
3. warms up for 8 sim s, then scores the scenario's duration: topic rates (sim time), real-time factor,
   whether every node is still alive, and outputs in the last 10 s;
4. stops the bag, the stack and the bridge in that order, so the bag and the JSON files are complete;
5. reads each evaluator's since-start summary (`<evaluator>.json`; every `*_eval_node` now has a
   `summary_file` parameter) and checks the metrics against the scenario's limits (`scenarios.py`).

Output: `~/carla_runs/suite_<time>/report.md` (all scenarios), `results.json`, and per scenario
`metrics.json`, `bridge.log`, `stack.log`, `gt_*.csv`, `bag/` (light set, ~1 MB/s: TF, odometry, IMU, GNSS,
ground-truth objects and boxes, perception objects / tracks / predictions, costmap). The exit code is 1 if
any scenario fails.

| Scenario | Scene | Stack | Checks |
|---|---|---|---|
| `static_props` | cones and boxes at 5-40 m, a parked car; ego drives past at 1 m/s | perception | static stays static while the robot moves, range, costmap |
| `crossing` | 6 pedestrians pacing across 5-30 m ahead, 0.8-1.4 m/s | perception | pedestrian range, tracking, 2 s prediction, dynamic risk |
| `occluded_crossing` | a van parked 6 m ahead on the left; 4 pedestrians step out from behind it | perception | tracking through occlusion, id switches, recall |
| `city` | autopilot, 40 cars, 80 pedestrians | perception | ground, car range, moving-car velocity, costmap, timing |
| `turns` | autopilot forced through Left, Right, Left, Right, Left at the next junctions | perception | ego-motion while turning: static cars stay static, flicker |
| `dropout` | city traffic; lidar silent 2 s at 30 s, camera 2 s at 50 s | perception | every node survives, outputs resume, costmap recall |
| `localization` | Stage 3 EKFs with the fixes on; autopilot, 10 cars | localization | global position and heading error |

New bridge options behind them: `--route Left,Right,...` (Traffic Manager turns at the next junctions),
`--dropout lidar:30:2,camera:50:2` (silence a sensor; ground truth keeps flowing so the evaluators see what
was missed), `--occluder` (with `--crossing`: the parked van). The `--scenario` parked car now tries six spots
and is lifted 0.3 m: at exactly ground height it never spawned.

**The limits** are the reference run below with a margin: a PASS means "no worse than when the limit was
set", not "good enough for the road". After a deliberate change that moves a number, update its limit in
`scenarios.py` and say why in the commit.

### Reference run (all PASS, 2026-10-21, `~/carla_runs/suite_reference`)

| Scenario | Key results |
|---|---|
| `static_props` | cones/boxes range MAE 0.24 m; static props' velocity 0.02 m/s while the ego drives; parked car 100 %, props 88 % covered by lethal cells; false obstacles 1.4 %; flicker 0.02 % |
| `crossing` | pedestrians: 82 % of boxes get a range, 99 % of them the right points, range MAE 0.15 m; tracking 0.09 m / 0.15 m/s; 2 s prediction 0.88 m, 95 % ellipse holds 90 %; costmap recall 99.2 %, dynamic risk holds 94 % of true positions 2 s ahead |
| `occluded_crossing` | through the van's occlusion: tracking 0.12 m, 0.2 id switches per 100 actor-frames; recall 97.5 %; dynamic risk 77 % at 2 s (people appearing from behind the van have no history yet) |
| `city` | ground precision / recall 95.5 / 99.8 %; car range MAE 1.27 m (far cars, see Stage 5); moving-car velocity 1.2 m/s; costmap recall 99.1 % moving / 100 % stopped; false obstacles 1.8 %; 45 ms p95 |
| `turns` | static cars' velocity 0.58 m/s while the ego turns; costmap recall 100 %; flicker 0.02 % |
| `dropout` | every node alive, outputs back after both dropouts, moving-car recall 100 % |
| `localization` | global RMSE 1.96 m (max 3.6 m) over 480 m, heading 2.0°; local EKF 3.9 m RMSE, 0.1 % drift |

**Repeatable.** The perception scenarios give the same numbers on every run with the same code: the city
run's ground precision, car range error and costmap recall matched to three decimals across three runs.
The global EKF is the exception (RMSE 0.56 / 0.56 / 1.96 m over three runs), because navsat_transform
takes its datum from whichever messages arrive first. Its limit is 3 m RMSE and 5 m maximum.

**Wall time** about 14 minutes for all seven. The city runs at 0.47× real time with 120 actors; the rest at 0.6-0.85×.
Light bags total about 1 GB.

### What the suite found

1. **Sidewalk pedestrians get almost no LiDAR hits, and fusion gave them the wall's range.** In the city
   run the LiDAR put at least 3 points on a pedestrian in only 23 actor-frames, all at 20-40 m, out of about 2,800
   camera boxes. The 16-beam LiDAR is pitched 15° down with its top beam level at 0.42 m, so beyond a few
   metres only that beam reaches. A pedestrian on a raised sidewalk shows it 0.27 m of shin, often behind
   the kerb. Fusion still found points in those boxes, the building or parked car behind, and reported
   their range: 0-3 % correct, 12-17 m too far. *Fix:* a size gate in `object_fusion.py`. A pedestrian box
   h px tall cannot be farther than about fy·1.95 m / h, so clusters and loose points beyond that are
   refused. Wrong pedestrian ranges fell from 17 % of boxes with a 12.4 m error to 11 % with a 6.8 m error.
   The crossing results are unchanged (99 % correct). What remains are objects at about the person's own
   depth, which no size test can separate.
   **For the rover:** in a city the camera is effectively the only pedestrian sensor beyond a few metres.
   Options: less LiDAR pitch, a monocular range from the box (size or ground contact) with a wide covariance, or
   a second, level LiDAR.
2. **The `--scenario` parked car never spawned.** It was placed exactly at ground height, which CARLA
   rejects as a collision. It is now lifted 0.3 m and tries six spots.
3. **Limits set from first guesses were wrong in the strict direction.** Ground precision is 95.5-97.4 % in
   every scene (Stage 4 measured 0.96-0.97), not ≥ 97 %, and the city's far cars make the car range MAE 1.27 m.
   The limits now come from the measured values.
4. **Not regressions, but visible in the numbers:** "static" pedestrians in the crossing scenes
   (`prediction.pedestrian_static.*`, `tracking.pedestrian.v_static_mae` ≈ 1 m/s) are the walkers turning
   round at the kerb, where the true speed passes through 0 while the track still carries the old velocity.
   City pedestrian prediction and dynamic risk can't be scored, because there are no tracks to score.

## Stages 10-11: ROS control and Nav2 to a GPS goal

```
GPS goal (lat, lon) --> gps_goal_node --> road route (CARLA lane graph, no traffic rules) --> sub-goal every 30 m
                                                                                                  | NavigateToPose
/costmap/nav (Stage 8: lethal obstacles + predicted risk) --> Nav2 costmaps --> Smac Hybrid-A* --> Regulated Pure Pursuit
/perception/predicted_objects (Stage 7) --> predictive_speed_limit_node --> /speed_limit ------------------^
Regulated Pure Pursuit --> /cmd_vel_nav --> collision_monitor (lidar points ahead of the nose) --> /cmd_vel
/cmd_vel --> carla_bridge --control ros (vehicle_control.py: speed PI, Ackermann steering, limits, watchdog) --> car
/carla/ego/collision (CARLA collision sensor): the referee
```

### Run it yourself

```bash
# Windows: CarlaUE4.exe.  WSL terminal 1: the car waits for /cmd_vel
ros2 run perceptron_carla carla_bridge --control ros                              # empty town
ros2 run perceptron_carla carla_bridge --control ros --traffic 30 --pedestrians 80   # with company
# WSL terminal 2: perception (Stages 4-8) + Nav2 + safety layers + gps_goal_node
ros2 launch perceptron_carla carla_nav.launch.py
# WSL terminal 3
rviz2 -d $(ros2 pkg prefix perceptron_carla)/share/perceptron_carla/rviz/carla.rviz
# WSL terminal 4: a goal
ros2 run perceptron_navigation gps_goal 19.1321586 72.9125739     # latitude longitude
ros2 run perceptron_navigation gps_goal --ahead 40                # or relative to the car
ros2 run perceptron_navigation gps_goal --cancel
```

In RViz, **Publish Point** (toolbar) on any spot sends the car there and logs that spot's GPS
coordinate; **2D Goal Pose** sends a plain Nav2 goal without the road route. Displays: *Nav2 plan*
(green), *GPS goal route* (white), *Pure pursuit carrot* (yellow), *Nav2 footprint*, *Collision monitor
stop / slow zone* (red / orange), and the Nav2 local / global costmaps (off by default).

| Piece | What it does | Why |
|---|---|---|
| `carla_bridge --control ros` | `/cmd_vel` (v, w) to throttle/brake/steer: curvature w/v becomes steering atan(L·k), a PI loop holds the speed, gear changes only at standstill | CARLA's Ackermann controller crashed the simulator (Stage 1) |
| safety in the bridge | speed ≤ `--max-speed` (4 m/s), steering ≤ `--max-steer-deg` (35°, turning radius 2.18 m), steer rate limit, brake if `/cmd_vel` is older than 0.5 s, `/carla/estop` (Bool) | a stalled or crashed planner must stop the car |
| collision sensor | `/carla/ego/collision` JSON per contact | the referee for the scenarios |
| `/costmap/nav` (costmap_node) | lethal obstacles + dynamic risk, **without** Stage 8's own inflation | Nav2 inflates with the car's real footprint |
| `config/nav2_carla.yaml` | Smac Hybrid-A* (Dubins, radius 3 m), Regulated Pure Pursuit 2.5 m/s with collision checking, rolling costmaps 30 m / 120 m, unknown = free | a car, not the rover's skid-steer base (its config is untouched) |
| `config/carla_nav_bt.xml` | replan at 2 Hz; recover by clearing the costmaps and waiting 3 s | Nav2's default tree spins in place; backing up is blind here (see findings) |
| `predictive_speed_limit_node` (perceptron_navigation) | Stage 7 predicted paths vs the corridor of the Nav2 plan: never reach a spot before a crossing person has cleared it, always able to stop 1.5 m short; `/speed_limit` | Nav2 replans *around* a walker, who then walks into the new path |
| `collision_monitor` (Nav2) | between the controller and the car: lidar points 0.2-1.8 m high in the 1.3 m ahead of the nose stop the car, in the 5 m ahead slow it to 40 %; no lidar for 1 s stops it | the last line, whatever the planner thinks |
| `gps_goal_node` (perceptron_navigation) | GPS to map through a datum; shortest road route over `/carla/map/road_graph` (any lane, either way, lane changes anywhere); NavigateToPose sub-goals 30 m apart; arrival within 3 m | goals far beyond the rolling costmaps |

### The four test conditions (suite scenarios, reference run 2026-10-22, `~/carla_runs/nav_reference`)

```bash
ros2 run perceptron_carla scenario_suite nav_open nav_cars nav_people nav_mixed     # ~8 min
```

| Scenario | Goal | Result | Time | Driven / route | Collisions | Closest pedestrian / car |
|---|---|---|---|---|---|---|
| `nav_open` (no cars, no people) | GPS 199 m by road | **PASS** | 80 s | 174 / 179 m | 0 | - / 7.8 m |
| `nav_cars` (30 cars) | GPS 166 m by road | **PASS** | 59 s | 117 / 136 m | 0 | - / 8.6 m |
| `nav_people` (6 crossing, 60 walking) | 60 m ahead, through the crossings | **FAIL in 2 of the last 4 runs** | 36-51 s | 58 / 60 m | 0 or 40 | 0.28 m (pass) / 0.00 (fail) |
| `nav_mixed` (30 cars, 80 people, 4 crossing) | GPS 198 m by road | **PASS** | 67 s | 144 / 163 m | 0 | 2.3 / 1.9 m |

Every run reached its goal; "driven" is shorter than "route" because the car cuts junction corners
(no traffic rules). Average speed is 2.0-2.2 m/s against the 2.5 m/s cruise.

**Open: nav_people.** In the failing runs, a crossing walker came at the car's front-right corner from 2-3 m
to the side, about 63° off the camera axis, outside the rover camera's 90° view. Objects are tracked and
predicted only when a camera box exists (Stage 5), so this walker had **no track and no prediction**. The
yield layer had nothing to yield to, and the car crept at 0.9 m/s into the walker's line. The first contact
(impulse 28) is the car's. The other ~39 events are the blind-steered walker pushing into the stopped car.
The fix belongs in perception, not Nav2: track moving lidar clusters that have no camera box, so the
predictions cover 360° and not only the camera's cone. Widening the collision monitor's zones instead would
stop the car beside every parked car in a narrow street.

### What building it found (each fixed, each with the number that showed it)

1. **Goals on the costmap edge.** A 40 m goal in an 80 m rolling global costmap is "out of costmap".
   Sub-goals reach 1.5 × 30 m, so the global window is 120 m.
2. **Unknown space is expensive for Smac.** As NO_INFORMATION it cost 3× per cell: a 40 m plan took 0.94 s
   (1.5 s timeouts at another spawn). Read as free, the same plan takes 0.01 s. The lidar shows obstacles as
   they come into its 20 m.
3. **Lanes only met at their ends.** Turning onto a street in the "wrong" lane meant driving to the next junction
   and back (a 433 m route for a 113 m goal). With lane-change edges (any lane within 4 m, +5 m cost) the route
   is 143 m and the drive took 60 s instead of 289 s.
4. **A 1.5 m arrival tolerance made the car circle.** A 3 m turning radius cannot null a 1-2 m side offset, so
   the car went round four times within 6 m of the goal. Arrival is now within 3 m, about one car length.
5. **Nav2 saw the pedestrians and drove into them anyway.** First nav_people run: 40 contact events. The bag shows
   every walker lethal in `/costmap/combined` at its true position for the 3 s before impact, and tracked as a
   moving pedestrian. But Smac replanned *around* each walker every 0.5 s, the walker walked into the new path,
   and Pure Pursuit only checks the next ~4 m of its arc, so it braked 0.5 s before contact.
   *Fix:* `predictive_speed_limit_node`, which yields to Stage 7's predicted crossings, plus Nav2's collision
   monitor. The run then had 0 collisions.
6. **Predictions end at 2 s, but crossing a car's corridor takes about 4 s.** The yield let the car in at the 2 s mark,
   it crawled into the crossing line, and a walker walked into its side (0.02 m). Paths are now extended at their
   final velocity until the walker leaves the corridor.
7. **Backing up is blind.** Nav2's BackUp recovery reversed into a pedestrian crossing behind the car (gap 0.00 m).
   The rover rig's lidar sits on the car's nose and the car body hides everything behind it. Recovery is now
   clear-and-wait only. In CARLA that is an artefact of the host car: the rover's own lidar sees 360°.
8. **Stuck at a junction corner for 80 s.** Pure Pursuit's 2.5-4 m constant-curvature arc cut through a pole that
   the plan went round, so it kept reporting "collision ahead". With a shorter carrot (1.5-6 m, 1.2 s), a 1 s
   collision horizon and plans kept further off obstacles (cost penalty 3, inflation 1.8 m), it no longer jams.
9. **Nav2 sometimes never became active.** A Humble DDS race lost a lifecycle service response ("failed to send
   response to /controller_server/change_state") when everything started at once under load. The lifecycle
   manager now starts 5 s after the servers.
10. **The dropout scene changed.** Adding the collision sensor changed CARLA's actor set, and Traffic Manager now
    routes the seed-4 ego differently (a red light by a junction). Moving-car recall there is 81.3 %, identical
    with and without the dropouts, so the dropout costs nothing; the limit is that scene's level. Why the costmap
    covers only 81 % of moving cars in that scene is open (the city scenario still gives 99 %).

**Known limits.** No traffic rules: the car uses any lane in either direction, ignores lights, and cuts
junction corners. The crossing scenes' walkers are steered blind, so a walker that turns at the kerb can
walk into the side of a car already passing. The suite counts those near passes (`nav.near_misses_*`, < 0.5 m)
and requires only "no touching" (> 5 cm) for pedestrians. Speed is 2.5 m/s, capped at 4 m/s by the bridge. The
rover itself is skid-steer: this CARLA Nav2 config is for the Microlino and is not the rover's.

## Commands

```bash
ros2 run perceptron_carla carla_bridge --help
ros2 run perceptron_carla carla_bridge --scenario                                # cones/boxes at 5-40 m
ros2 run perceptron_carla carla_bridge --traffic 40 --pedestrians 60 --autopilot # busy city
ros2 launch perceptron_perception carla_overlay.launch.py                        # lidar overlay + check
ros2 run perceptron_carla sensor_check
```
