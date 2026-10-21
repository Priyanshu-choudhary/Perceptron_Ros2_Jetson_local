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

## Commands

```bash
ros2 run perceptron_carla carla_bridge --help
ros2 run perceptron_carla carla_bridge --scenario                                # cones/boxes at 5-40 m
ros2 run perceptron_carla carla_bridge --traffic 40 --pedestrians 60 --autopilot # busy city
ros2 launch perceptron_perception carla_overlay.launch.py                        # lidar overlay + check
ros2 run perceptron_carla sensor_check
```
