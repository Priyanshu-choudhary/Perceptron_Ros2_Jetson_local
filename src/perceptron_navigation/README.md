# perceptron_navigation

LiDAR, SLAM, localisation, Nav2 and waypoint missions. This is the package that
turns the robot from "drives where you tell it" into "goes where you ask it".

```
config/
  slam_toolbox.yaml    mapping parameters (level 2)
  nav2_params.yaml     AMCL, costmaps, planner, controller, behaviours (levels 3-7)
  waypoints.yaml       the patrol route
launch/
  slam.launch.py           slam_toolbox on its own (works on hardware too)
  navigation.launch.py     Nav2 + localisation + cmd_vel bridge
  nav_simulation.launch.py everything, in the Gazebo room
perceptron_navigation/
  cmd_vel_relay.py     Nav2's Twist -> the base's topic, with an on/off service
  mission_node.py      patrol state machine, hands over to ArUco docking
  mission_client.py    `ros2 run perceptron_navigation mission start`
maps/                  saved occupancy grids (empty until you make one)
rviz/navigation.rviz   map, costmaps, plans, scan, dock estimate
```

---

## Status

Be honest with yourself about what has actually been run.

| level | what it is | state | evidence |
| --- | --- | --- | --- |
| 1 | LiDAR to obstacle visualisation | **verified** | `tools/lidartest.sh`: 500 pts, 360 deg, mean wall-range error 2 mm |
| 2 | LiDAR + encoders + IMU to SLAM map | **verified** | `tools/navtest.sh slam`: 99% of the room mapped, map saved |
| 3 | Localisation on a saved map | **verified** | `tools/navtest.sh localize`: converged to 0.115 m / 0.9 deg from a seed 0.28 m and 9.7 deg off |
| 4 | NavigateToPose | **verified** | `tools/navtest.sh goal`: 3/3 goals, ground-truth error 0.096 to 0.217 m |
| 5 | Static obstacle avoidance | **verified** | same test: routed around the partition instead of through it |
| 6 | Dynamic obstacle avoidance | config live, **not separately tested** | the obstacle layer is enabled in both costmaps, but the world has nothing that moves |
| 7 | Waypoint missions | see `tools/navtest.sh mission` | |
| 8 | Autonomous exploration | not started | see "Where this goes next" |
| 9 | Camera and LiDAR fusion | not started | |
| 10 | Autonomous patrol, docking, missions | see `tools/navtest.sh mission` | |

Prerequisites, if you are setting this up somewhere new:

```bash
sudo apt update && sudo apt install -y \
    ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
    ros-humble-nav2-simple-commander ros-humble-robot-localization
```

Level 1 needs none of them.

---

## Testing everything

Four headless tests, each measured against Gazebo ground truth rather than
against the robot's own belief about where it is. Run them in this order,
because everything after `slam` needs the map it saves.

```bash
bash tools/lidartest.sh          # level 1     is the scanner mounted right
bash tools/navtest.sh slam       # level 2     wander, map, save
colcon build --symlink-install --packages-select perceptron_navigation
bash tools/navtest.sh localize   # level 3     AMCL convergence
bash tools/navtest.sh goal       # levels 4-6  NavigateToPose around the partition
bash tools/navtest.sh mission    # levels 7,10 patrol then dock
```

Budget three to twelve minutes each; most of that is Gazebo and Nav2 starting
up. [`tools/README.md`](../../tools/README.md) explains how to read the output.

Two things that will otherwise waste an afternoon:

**Nav2 takes about two minutes to reach `active` from `/mnt/c`.** Lifecycle
activation reads a great many small files and the Windows filesystem bridge is
slow at that; on the Linux filesystem it is closer to fifteen seconds. If goals
come back rejected, check the lifecycle state before anything else:

```bash
ros2 lifecycle get /bt_navigator
```

**Waiting for the action server is not the same as waiting for the node.**
`bt_navigator` creates `navigate_to_pose` during *configure*, so
`wait_for_server` succeeds — and every goal is rejected — until activation
actually finishes.


---

## The hardware: LDROBOT D500

The D500 kit is the LD19 / STL-19P DTOF scanner.

| | |
| --- | --- |
| Range | 0.03 to 12 m, ±45 mm |
| Scan rate | 10 Hz |
| Sample rate | 5000 Hz, so ~500 points per revolution |
| Field of view | 360° |
| Ambient light | up to 30 kLux |
| Interface | UART, usually over a CH340 USB adapter |

The simulated sensor in `perceptron_robot.gazebo` uses exactly these numbers, so
what you tune against in simulation should transfer.

### Driver on the real robot

The D500 is a DTOF unit, so it is the `ldlidar_stl_ros2` driver, not the
triangulation one:

```bash
cd ~/ros2_ws/src
git clone https://github.com/ldrobotSensorTeam/ldlidar_stl_ros2.git
cd .. && colcon build --symlink-install --packages-select ldlidar_stl_ros2
sudo chmod 777 /dev/ttyUSB0        # or a udev rule, see below

ros2 launch ldlidar_stl_ros2 ld19.launch.py
```

Three parameters have to agree with this workspace:

| driver parameter | value | why |
| --- | --- | --- |
| `product_name` | `LDLiDAR_LD19` | the D500 is an LD19 |
| `laser_scan_topic_name` | `scan` | what Nav2 and slam_toolbox subscribe to here |
| `frame_id` | `laser_link` | must match the URDF, or the map comes out rotated against the robot |

The driver defaults `frame_id` to `base_laser`. **Change it to `laser_link`** or
add a static transform; a mismatch here is the single most common reason a first
SLAM run produces a smeared, rotated map.

A udev rule beats `chmod 777` every time:

```bash
echo 'KERNEL=="ttyUSB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE="0666", SYMLINK+="ldlidar"' \
  | sudo tee /etc/udev/rules.d/99-ldlidar.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Then use `/dev/ldlidar` and the port stops moving around.

### Where it is mounted

`perceptron_robot.xacro` puts `laser_link` at **0.21 m above the ground, over
the robot's centre of rotation** (`lidar_x/y/z` near the top of the file).

Two reasons, both worth keeping when you bolt the real one on:

- **Over the centre of rotation.** Turning on the spot then produces no lidar
  translation, which keeps scan matching well conditioned. An off-centre
  scanner is the usual reason a map smears every time the robot spins.
- **Above the camera.** The camera body tops out at 0.1821 m. At 0.21 m the
  scan plane clears it by 28 mm; any lower and the robot's own camera would
  carve a permanent blind wedge out of every scan.

If you mount it somewhere else, change `lidar_x`, `lidar_y`, `lidar_z` and
re-run `bash tools/lidartest.sh`. Anything but a near-zero mean error means the
URDF and reality disagree.

---

## Level 1 — see the room

```bash
ros2 launch perceptron_robot_bringup gazebo_control2.launch.py \
    world_path:=$(ros2 pkg prefix perceptron_robot_gazebo)/share/perceptron_robot_gazebo/worlds/room_world.world \
    x:=-3.0 y:=-2.0
```

Add a **LaserScan** display on `/scan` in RViz (Reliability: Best Effort). Or
check it numerically:

```bash
bash tools/lidartest.sh
```

That spawns the robot at a known pose and compares measured ranges against the
distances to the walls computed from Gazebo ground truth. Reference output:

```
scan  frame_id : laser_link
      points   : 500        angle range: -3.142 to +3.142 (360.0 deg span)
      rate     : ~9.7 Hz    range limits: 0.05 to 12.00 m
wall-only bearings: 20 of 24, mean error +0.0020 m, sd 0.0136 m
```

The sd is the 15 mm noise in the sensor model. A **mean** that is not near zero
means the mount pose is wrong; an error that grows with bearing means
`laser_link` is rotated.

---

## Level 2 — build a map

```bash
ros2 launch perceptron_navigation nav_simulation.launch.py slam:=true
```

Drive around with teleop until the room closes up in RViz:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args -r /cmd_vel:=/diff_drive_controller/cmd_vel_unstamped
```

Drive **slowly**, and cover each wall from more than one angle. Then save:

```bash
ros2 run nav2_map_server map_saver_cli -f src/perceptron_navigation/maps/room_map
colcon build --symlink-install --packages-select perceptron_navigation
```

What SLAM is actually doing: `slam_toolbox` matches each new scan against the
ones already in its pose graph, using wheel odometry as the initial guess. When
it recognises somewhere it has been before it closes the loop and corrects the
whole graph at once. It publishes the `map -> odom` transform, which is the
correction between where odometry thinks the robot is and where it really is.

That is why the drivetrain calibration from
[`perceptron_robot_control`](../perceptron_robot_control/README.md) matters
here: odometry is the prior for every scan match. Bad odometry means SLAM works
harder and drifts sooner.

Parameters worth knowing in `config/slam_toolbox.yaml`:

| parameter | default | effect |
| --- | --- | --- |
| `resolution` | 0.05 | Map cell size, metres. |
| `max_laser_range` | 10.0 | Below the sensor's 12 m: far returns are noisy and hit walls at glancing angles. |
| `minimum_travel_distance` / `_heading` | 0.2 / 0.25 | How far the robot moves before another scan enters the graph. Tighter is a better map and more CPU. |
| `loop_search_maximum_distance` | 3.0 | How far away a loop closure is looked for. |
| `mode` | `mapping` | `localization` re-uses a saved pose graph instead of a map file. |

---

## Level 3 — localise on the map

```bash
ros2 launch perceptron_navigation nav_simulation.launch.py slam:=false
```

AMCL replaces slam_toolbox: a particle filter that keeps a cloud of pose
hypotheses, weights them by how well the current scan matches the stored map,
and resamples. It needs a starting hint — **2D Pose Estimate** in RViz, or spawn
the robot where you started mapping.

Symptoms and causes:

| symptom | cause |
| --- | --- |
| Particle cloud never converges | Initial pose too far off, or the map does not match the world. |
| Robot jumps around the map | `alpha1..4` too high, or odometry is genuinely bad. |
| Pose drifts and never corrects | Not enough geometric features; a long blank corridor looks the same everywhere. |

---

## Levels 4 to 6 — Nav2

Same launch as level 3. Give a goal with **2D Goal Pose** in RViz, or:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 2.0, y: 2.0}, orientation: {w: 1.0}}}}"
```

What the pieces do:

```
   NavigateToPose action
          |
      bt_navigator        runs a behaviour tree: plan, follow, recover, repeat
          |
   +------+---------------------------+
   |                                  |
planner_server (NavFn)         controller_server (DWB)
   global path over the           picks (v, w) each 50 ms by scoring
   global costmap                 candidate trajectories on the local costmap
   |                                  |
global_costmap                   local_costmap
   static layer   = the saved map     obstacle layer = live /scan  (LEVEL 6)
   obstacle layer = live /scan        inflation layer
   inflation layer                    4 x 4 m rolling window
   (LEVEL 5)
```

- **Level 5, static obstacles** is the global costmap's static layer: the walls
  and crates you mapped, inflated by `inflation_radius` so the planner keeps its
  distance.
- **Level 6, dynamic obstacles** is the obstacle layer in *both* costmaps. It
  marks what the lidar sees now and clears what it does not, so something that
  moves into the robot's path appears in the local costmap, DWB scores its
  trajectories as fatal, and the behaviour tree replans. Nothing extra to
  enable — it is on.

Robot-specific numbers in `config/nav2_params.yaml`:

| parameter | value | why |
| --- | --- | --- |
| `robot_radius` | 0.33 | The chassis is 0.415 × 0.4526 m and `base_footprint` is not quite central, so the furthest corner is 0.319 m away. A robot that turns on the spot is honestly modelled by its circumscribed circle. |
| `inflation_radius` | 0.55 | A little over the radius: prefer the middle of a gap, but still fit through one. |
| `max_vel_x` | 0.35 | Well under the drivetrain's 0.6 m/s so the skid-steer is not at its friction limit while planning. |
| `max_vel_theta` | 1.0 | Under the 1.5 rad/s limit, same reason. |
| `min_vel_x` | 0.0 | **No reversing.** There is no rear-facing sensor, so backing up under the planner's control is driving blind. The recovery behaviours can still back up deliberately. |
| `xy_goal_tolerance` | 0.15 | Nav2's job is to get close. Precision at the dock is the ArUco controller's job. |
| `obstacle_max_range` | 8.0 | Below the sensor maximum; far returns are the noisiest and marking them smears the costmap. |

### cmd_vel_relay

Nav2 publishes `Twist` on `/cmd_vel`, or `/cmd_vel_smoothed` when a
velocity_smoother is in the stack (Humble runs one). The simulated base listens
on `/diff_drive_controller/cmd_vel_unstamped`; the real base listens on
`/cmd_vel`. `cmd_vel_relay` bridges the two and, more importantly, can be
switched off:

```bash
ros2 service call /cmd_vel_relay/enable std_srvs/srv/SetBool "{data: false}"
```

That is how the mission node hands the base over to the docking controller
without the two fighting. If the relay warns that it has seen nothing, your Nav2
is publishing on the other topic:

```bash
ros2 topic list | grep cmd_vel
ros2 launch perceptron_navigation navigation.launch.py nav_cmd_vel_topic:=/cmd_vel
```

---

## Levels 7 and 10 — the patrol mission

```
Start -> WP1 -> inspect -> WP2 -> inspect -> WP3 -> inspect -> home -> dock
```

```bash
ros2 launch perceptron_navigation nav_simulation.launch.py slam:=false
ros2 run perceptron_navigation mission start
ros2 run perceptron_navigation mission cancel
ros2 topic echo /mission/status
```

The route is `config/waypoints.yaml`, in the `map` frame. To collect real
coordinates, drive the robot to a spot and read them off:

```bash
ros2 run tf2_ros tf2_echo map base_footprint
```

`inspect` is a placeholder that holds position for `inspect_seconds` — long
enough for the camera to get clean frames. Replace the body of `_inspect()` in
`mission_node.py` with the actual task and keep the cancel check.

### The handover to docking

This is the interesting part, and the reason level 10 is not just "level 7 plus
a dock command".

Nav2 gets the robot to a **staging pose** about 1.2 m in front of the dock, then
the ArUco controller takes over. They cannot both drive:

```
mission_node:  goToPose(dock_staging)          Nav2 driving
               /cmd_vel_relay/enable false     Nav2 muted
               /docking/start                  ArUco controller driving
               wait for /docking/status DOCKED
               /cmd_vel_relay/enable true      Nav2 driving again
```

Without the mute, Nav2 keeps issuing goal-reached corrections while the docking
controller is creeping the last few centimetres, and the robot oscillates in
front of the dock.

Why not let Nav2 dock? Its goal tolerance is 0.15 m and it localises against a
map, while docking needs about 0.02 m relative to a marker the robot can
actually see. Different sensor, different frame, different job. Map-frame
localisation is also exactly the thing that drifts; the marker does not.

---

## AR path overlay — the plan drawn on the camera

`/plan` is projected into the front camera's pixels and drawn on the live video,
so the operator sees where the robot intends to go on the floor of the actual
room rather than on a map beside it.

```bash
# on the Jetson, in two shells
python3 jetson_robot_bridge.py --no-aruco        # releases /dev/video0
python3 jetson_path_overlay.py --host-ip <operator>

# on the ROS host, alongside navigation.launch.py
ros2 launch perceptron_navigation path_overlay.launch.py

# on the operator machine
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H265, payload=96" ! rtph265depay ! h265parse ! avdec_h265 ! autovideosink sync=false
```

### Why the pixels are computed here and drawn there

The camera never joins the ROS graph — an uncompressed 1280x720 stream is
~41 MB/s and `perceptron_robot_description/README.md` records what that does to
node discovery on a 4 GB WSL host. The frame cannot come to the geometry, so the
geometry goes to the frame: a decimated path is ~2.8 kB, about 41 kB/s at 15 Hz.

`path_overlay_node.py` owns TF, the intrinsics and the Nav2 topics.
`jetson/jetson_path_overlay.py` receives a list of shapes and draws them,
knowing nothing about ROS. Projection bugs stay on the machine with a debugger.

**Nothing in this path steers the robot.** If the link dies the operator loses a
drawing. That is the whole failure mode.

### The camera geometry this depends on

Everything rests on `base_link -> camera_link` in the xacro being right:
lens 0.42 m above the floor, pitched `-0.07027` rad (**up** 4.03°). Those were
measured, not modelled, and the overlay is brutally sensitive to them — a 1°
pitch error puts the path ~9 cm off at 5 m, which reads as a ribbon floating
above the floor or sinking into it. That is the best check there is that the
extrinsics are still true: if the ribbon does not lie flat on the ground, go and
re-measure the mount before believing anything else.

Consequences of that geometry, all verified against the arithmetic:

| | |
| --- | --- |
| horizon | v = 377 |
| nearest visible floor | 0.535 m ahead of the lens |
| floor occupies | v = 377 … 720 |
| 2 m / 5 m / 8 m ahead | v = 465 / 411 / 398 |

5 m and 8 m are **13 px apart**, which is why `max_range` is 8.0: past that the
path is pixels from the horizon and contributes nothing but jitter.

### Why the cull is two-stage and not just Z > 0

The lens is ~115° horizontal (fx ≈ 411 over 1280 px) and a 5-coefficient
plumb-bob model is only valid inside the cone it was fitted in. Differentiating
the radial polynomial with this calibration:

```
r = 1.761  image corner  (60.4° off axis)   d/dr = +1.07
r = 2.653  model stops being monotonic (69.3°)
r = 3.0    outside the FOV                  d/dr = -1.29
```

Past r ≈ 2.65 the model **folds**: points well outside the field of view come
back with plausible pixel coordinates inside the image. Sweeping ground points
across bearings, 37 of them land inside the frame — e.g. a point 73° off axis
projecting to (1252, 482). The symptom would be a phantom stripe whipping across
the picture whenever the robot turns and the path sweeps past the lens edge, and
`Z > 0` does not catch it because those points are genuinely in front of the
camera, just not in view.

So the mask is `Z > min_z` **and** `r <= 2.35` (66.9°), applied before
`projectPoints`. That sits above the image corner and below the fold. It also
bounds the projected pixel magnitude, which is what stops the int32 overflow an
unculled point produces — an 84°-off-axis point projects to (3707183, -544221).

A polyline is only drawn between two points that both survived; joining across
a gap draws a chord through exactly what the cull was protecting against.

### The ribbon

The band is the chassis's true width (0.4526 m), offset **in metres on the
ground plane** and then projected — not offset in pixels, which would draw a
constant-width screen stripe that reads as a flat sticker. Perspective narrows
it with distance (193 px at 1 m, 47 px at 4 m), which is what sells it as lying
on the floor, and it doubles as a clearance gauge: if it fits between two
obstacles on screen, the robot fits.

### Clocks

**No timestamp is compared across the two machines.** The host's and the
Jetson's clocks are independent and undisciplined — `JetsonClock` in
`jetson_bridge_node.py` exists solely because of that. Overlay age is measured
on the Jetson as `time.monotonic()` since it received the payload.

The TF lookup asks for the *latest* transform rather than the frame's capture
time, because the frame is on the far side of a UDP video link with no shared
clock. `aruco_detector_node.py` already does the same. The cost is that while
the robot moves, the overlay lags the picture by the video pipeline's latency
and appears to swim slightly. That is cosmetic and expected, not a projection
bug; past 0.35 s the renderer says so on the HUD, and past 0.7 s it drops the
overlay rather than show a path frozen where the robot used to be going.

## Where this goes next

**Level 8, exploration.** Add `nav2_wfd` or `explore_lite`: pick frontiers
between known-free and unknown cells, send each as a `NavigateToPose` goal,
repeat until none are left. The whole level-2 stack already runs; this just
replaces you-with-a-keyboard as the source of goals.

**Level 9, camera and LiDAR fusion.** The natural first step is not fusion but
*layering*: put the ArUco detections into the costmap as a semantic layer, or
use the camera to classify what the lidar found. True fusion (projecting depth
into the scan, or a shared occupancy representation) only pays off once there is
something the lidar genuinely cannot see — glass, negative obstacles, low
objects under the 0.21 m scan plane. **That last one is worth taking seriously
now**: anything shorter than 0.21 m is invisible to the planner, and a 2D lidar
robot will drive straight into a doorstep it cannot see.

**Level 10.** The mission node is the skeleton. What it still lacks is battery
awareness (dock when low, not just at the end of a route), mission persistence
across restarts, and a way to schedule patrols.

---

## Troubleshooting

Everything here was hit while bringing this stack up, in this world, on this
machine. The pattern worth internalising: Nav2 rarely tells you what is wrong
in the message you first see. Work down the stack — lifecycle, then transforms,
then costmap, then the controller.

| symptom | check |
| --- | --- |
| `/scan` missing | Is the ray sensor in the URDF? `ros2 topic list \| grep scan`, then the gzserver log. In simulation it must be `type="ray"`, not `gpu_ray`, under WSL. |
| Map smeared or rotated | `frame_id` mismatch between the driver and `laser_link`, or the lidar mounted somewhere other than where the URDF says. Run `bash tools/lidartest.sh`. |
| Map drifts on turns | Lidar not over the centre of rotation, or `wheel_separation_multiplier` uncalibrated. |
| Nav2 nodes never activate | `ros2 lifecycle get /bt_navigator`. Bringup takes about two minutes from `/mnt/c`. If it is stuck on `controller_server`, the robot did not exist when Nav2 started, so the local costmap has no `odom -> base_link`. |
| Every goal rejected | Almost always the lifecycle again. `bt_navigator` creates `navigate_to_pose` during *configure*, so the action server answers and rejects everything until activation finishes. |
| `The goal sent to the planner is off the global costmap` | You are in SLAM mode and the goal is in unexplored space. The global costmap only covers ground already seen. Teleop while mapping; navigate once you have a map. |
| Robot does not move on an accepted goal | `ros2 topic info /cmd_vel --verbose`. In Humble the chain is `controller_server -> /cmd_vel_nav -> velocity_smoother -> /cmd_vel`, so the *end* of it is `/cmd_vel`. If `cmd_vel_relay` warns it has seen nothing, its `input_topic` is wrong. |
| `Failed to make progress`, then a backup recovery | `progress_checker` wants `required_movement_radius` within `movement_time_allowance`. On a loaded machine a legitimate tight manoeuvre misses it. Raised to 25 s here. |
| `Behavior Tree tick rate 100.00 was exceeded!` | The machine cannot hold the BT tick. Raise `bt_loop_duration` and `default_server_timeout`. Left unaddressed this shows up as action calls timing out and goals failing for no stated reason. |
| A mission waypoint always fails, others work | `python3 tools/checkwaypoints.py`. A goal closer to an obstacle than `robot_radius` is unreachable by construction; one inside `inflation_radius` is reachable but expensive. |
| A journey is cancelled at a suspiciously round time | Something is timing out on the **wall clock** while the robot moves on the **simulation clock**. At a real-time factor of 0.5, a 180 s allowance is 90 s of robot time. Every timeout in this workspace uses the node clock; check anything you add. |
| Planner takes a wide berth | `inflation_radius` and `cost_scaling_factor`. |
| Oscillates near goals | `xy_goal_tolerance` too tight for the base, or DWB's `RotateToGoal` scale too high. |
| Robot refuses to plan after docking | It is parked against the dock, deep inside inflated space. Clear the costmap or drive out manually: `ros2 service call /global_costmap/clear_entirely_global_costmap nav2_msgs/srv/ClearEntireCostmap`. |
| `/map` never arrives at your node | It is latched (TRANSIENT_LOCAL) and published once. A VOLATILE subscriber that joins later gets nothing. slam_toolbox hides this by republishing; `map_server` does not. |
| AR overlay: video fine, nothing drawn | Read the HUD, which always says which case it is. `no plan - planner not running` means nothing publishes `/plan`, so Nav2 is not up. `no plan - idle, send a goal` means Nav2 is up and waiting. `no tf <frame> -> camera_optical_link` means the transform chain does not reach the camera — usually the robot bringup is not running, so there is no `odom` either. `plan outside camera view` means it projected but every point failed the cull. |
| AR overlay: Jetson says `0 payloads` | The host node is not running or cannot reach the Jetson. `0 payloads` with frames still flowing is the renderer working correctly on an empty inbox. The host logs `N overlay frames dropped and none sent` when it is pushing into the void. |
| AR overlay: path lags the picture | Expected while moving: the TF lookup is `Time()` (latest) because the video crosses a link with no shared clock. The HUD says `overlay lagging N ms` past 350 ms and the overlay is dropped past 700 ms. |

## GPS goals and predictive yielding (from the CARLA Stage 11 work)

Two nodes here are robot-side and were built and tested in CARLA (see perceptron_carla's README, Stages 10-11):

* `gps_goal_node` + `gps_goal` CLI: drive to a latitude/longitude. GPS becomes map coordinates through a datum
  (`datum_lat`, `datum_lon`, `datum_frame`; on the rover, navsat_transform with a fixed datum makes `map` that
  frame). The node routes straight or over a road graph (`route_mode: roads`), and hands Nav2 `NavigateToPose`
  sub-goals 30 m apart. It also accepts RViz *Publish Point* clicks and logs their GPS. Pure logic: `geo.py`.
* `predictive_speed_limit_node`: publishes a `/speed_limit` for controller_server from the Stage 7 predictions,
  so the robot doesn't reach a spot on its plan before a crossing person has cleared it, and can always stop
  short of it. Pure logic: `yielding.py`.

Neither is in the rover's launch files yet: the rover's `nav2_outdoor_gps.yaml` and its launch are untouched.
