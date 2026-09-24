# perceptron_edge — the whole stack on the Jetson Nano

SLAM mapping, AMCL + Nav2 navigation, optional EKF, ArUco pose seeding and
CT6B teleop all run on the robot's own computer, headless. The laptop only
*looks* at the robot, through Foxglove.

## Every day

```bash
ssh jetson@<jetson-ip>
~/robot start slam                  # build a map; drive with Foxglove Teleop or goals
~/robot save-map kitchen            # save it
~/robot stop
~/robot start nav map=kitchen       # navigate on it
~/robot pose 1.0 -0.5 2.58          # or "Publish pose estimate" in Foxglove
~/robot goal 2.0 0.5 0              # drive there; prints progress, Ctrl-C cancels
~/robot dock -0.341 0.0477 0        # go charge (see Charging dock below)
~/robot undock                      # back off the charger
~/robot status                      # what is running, battery
~/robot cpu                         # what each part costs
~/robot bridge                      # serial bridge only: run ROS on the laptop instead
```

`robot` asks for the sudo password once (it mounts the chroot); ROS itself
runs as user jetson, never as root.

On the laptop, open Foxglove (the Windows app, or app.foxglove.dev in Chrome):
**Open connection → Foxglove WebSocket → `ws://<jetson-ip>:8765`**.

`robot goal` and `robot pose` go through Nav2's action / wait for AMCL rather
than `ros2 topic pub /goal_pose`: bt_navigator takes `/goal_pose` best-effort
and Foxglove subscribes to it too, so a one-shot publish could reach Foxglove
and not Nav2 - the goal that "needed sending 2-3 times".

## Charging dock

The charger stands against the wall with the 2x2 ArUco board (DICT_6X6_250,
ids 0-3) above it. Nav2 cannot reach the contacts (they are inside the wall's
inflation, and need +-3 cm), so:

1. `robot dock X Y YAW` sends Nav2 to X Y YAW, a spot about 35 cm in front of
   the charger (map frame; `-0.341 0.0477 0` for the room map). The camera is
   off during this leg.
2. The camera switches on. The robot lines up on the charger's axis 22 cm out,
   drives in slowly, checks its line again 10 cm out, and stops when the
   battery current turns negative.
3. No charge? Push 1 cm, wiggle, back off, line up again aiming 1.2 / 2.4 cm to
   either side - up to 5 attempts. It prints every step; Ctrl-C or
   `robot cancel` stops the robot.

| command | |
|---|---|
| `robot dock X Y YAW` | the whole thing; X Y YAW is remembered |
| `robot dock` | the same with the remembered X Y YAW |
| `robot dock here` | camera part only, robot already facing the charger |
| `robot undock [M]` | reverse M m (0.35) off the charger, watching behind with the lidar |
| `robot dock-watch` | live: board distance, offset from the docked pose, current. Never moves. |
| `robot dock-teach` | **once, and whenever the charger or camera moves**: robot on the charger and charging. Records the docked view, reverses 25 cm to calibrate the camera tilt, drives back on. |

The taught pose is in `/opt/perceptron_ws/config/dock_station.yaml` (in the
chroot), backed up in this repo as `config/dock_station.yaml`; tuning is
`config/dock.yaml`. The room map it was taught on is
`perceptron_navigation/maps/room.{yaml,pgm}` (the robot keeps its maps in
`~/robot_maps`). Progress is also on `/dock/status` for
Foxglove.

**Goals given while on the charger** (Foxglove's goal tool, `robot goal`,
anything that uses Nav2) back off the charger first, then navigate. Without
that, Nav2 accepts the goal (the planner finds a path) but every motion - the
rotation shim, DWB, Spin, BackUp - refuses to start from a footprint that
already touches the dock ("Collision Ahead - Exiting DriveOnHeading"), so the
robot sat still and the goal failed after its recoveries. The navigate-to-pose
tree (`behavior_trees/navigate_to_pose.xml`) now starts with
`UndockIfOnCharger`, answered by the `dock_guard` node: instantly when the
robot is not docked, otherwise after `robot undock`'s 0.35 m reverse (lidar
watching behind). Docked = `/battery/state` CHARGING, or AMCL within 12 cm of
the taught dock pose. Measured: goal from the charger to 1.2 m away, ~15 s
backing off, then 11 s to arrive. `robot logs` shows "goal while on the
charger ... backing off".

**`/battery/state` current is now positive while charging** (ROS convention;
its status field says CHARGING). The sensor reads the other way round, which
`battery_real.yaml` now corrects with `measured_current_sign: -1`. The raw
sensor value, negative on the charger, is still on `/battery/measured_current`.

Measured 24 Sep 2026 (`robot dock -0.341 0.0477 0` from 1.2 m away, facing
away): Nav2 10 s, docked and charging on the first attempt, 45 s in all. From
35 cm in front: 16 s. With a deliberate 5 cm miss first: charging on attempt 3.

Good to know:

* **Fix the charger box to the floor.** On every miss the robot drove 2.3 cm
  past the taught pose without stalling, and after an off-centre touch the
  contacts were no longer where teach found them.
* **The camera looks 5.2 deg lower than the URDF says** (teach measured it
  from a straight reverse; vision and odometry agreed on distance to 0.2%).
  Docking corrects for it itself, but `aruco_localizer_node` and the path
  overlay use the URDF and are off by that much.
* The dock board has the same ids as the `wall_north` board in
  `marker_map.yaml`. If it is the same sheet, moved, the marker map is stale -
  do not start with `aruco=true` until it is re-surveyed.
* **Check the contacts for a short.** In one test the whole robot (Jetson and
  STM32) lost power at the instant it pressed onto the contacts, battery at
  11.6 V. The attempt before had touched with the current falling from 1.4 A
  to 0.04 A without going negative - contact, but not a clean one. Look for a
  pad that can bridge + and - when the nose arrives off-centre, and for
  polarity.
* The STM32 USB link wedges now and then under motor load, more often on a low
  battery; the bridge reopens it in ~5 s and docking stops and waits.
* The camera costs nothing until docking: the bridge runs with
  `--aruco-on-demand` and opens it only while `dock` asks (~0.4 core at 10 Hz).

## Options for `robot start`

| option | default | what it does |
|---|---|---|
| `map=NAME` | `room_map` | nav mode: a map from `robot maps`, or a `.yaml` path |
| `ekf=true` | false | EKF (wheel + IMU) owns odom → base_footprint |
| `aruco=true` | false | camera marker detection + `aruco_localizer_node` seed AMCL. Nothing ArUco-related runs unless this is set. |
| `nav=false` | true | slam mode without Nav2 |
| `teleop=true teleop_port=/dev/ttyUSB2` | off | CT6B RC receiver on the Jetson. Give the port explicitly: ttyUSB0/1 are the LiDAR and the STM32. |
| `motors=false` | true | bench mode: everything runs, the motors ignore /cmd_vel |
| `foxglove=false` | true | no laptop view |
| `log_level=info` | warn | Nav2 / SLAM verbosity |

## How it is put together

* **ROS 2 Humble** is installed in an Ubuntu 22.04 chroot at
  `/mnt/sdcard/ros2_chroot`, because JetPack 4.6 is Ubuntu 18.04.
* **This workspace** lives at `/opt/perceptron_ws` inside the chroot. The maps
  are at `~/robot_maps` on the host.
* **`robot`** (`host/robot`) runs on the host. It:
  * mounts the chroot;
  * keeps `~/jetson_robot_bridge.py` running, camera always on for `aruco=true`,
    otherwise `--aruco-on-demand` (closed until docking asks for it);
  * runs `ros2 launch perceptron_edge edge.launch.py` inside the chroot as
    user jetson.
* **ROS traffic stays on the robot**: Fast DDS over UDP on 127.0.0.1 only
  (`config/fastdds_localhost_udp.xml`, set by `robot`). Not
  `ROS_LOCALHOST_ONLY=1`: in Humble that also turns on Fast DDS shared memory,
  which on this Nano got stuck after any short-lived ROS command (`robot dock`,
  `goal`, `pose`, `dock-watch`) left the graph - every node spun re-sending to a
  full queue and the idle stack went from 1.06 to 3.8 of 4 cores within 30 s
  until ROS restarted. On UDP: 1.04 cores after five such commands in a row.
  `foxglove_bridge` is the only way in: topic whitelist in
  `config/foxglove.yaml`, `/scan` offered as `/viz/scan` at 5 Hz, and nothing
  is subscribed while nobody watches.

## Measured on the robot (bench mode, 23 Sep 2026)

| state | ROS nodes | whole Nano (of 4 cores) | memory |
|---|---|---|---|
| nav, localised, idle | 0.33 cores | 0.90 | 364 MB |
| nav, goal active (DWB at 10 Hz) | 0.67 cores | 1.05 | 400 MB |
| slam + Nav2, parked | 0.55 cores | 1.04 | 356 MB |
| + Foxglove streaming to the laptop | +0.10 | 1.11 | +20 MB |
| nav with ekf=true aruco=true | 0.97 cores | 1.28 | 601 MB |

*Whole Nano* includes `jetson_robot_bridge.py` (~0.35 cores) and the Ubuntu
desktop. `aruco_localizer_node` alone costs ~26% of a core even with no
camera: seed with `aruco=true`, then `robot restart` without it.

## What was optimised

Measured with `tools/perf`, 23 Sep 2026. On the Nano, the stock laptop
configuration used 2.74 of 4 cores while navigating and missed 299 control
deadlines per 30 s. The optimised one used 0.52 cores and missed 2.

The largest wins, none of which change navigation behaviour:

* no RViz, and no ArUco node unless asked;
* lifecycle bonds off;
* `jetson_bridge_node` publishing odom/TF at 50 Hz, battery at 2 Hz and IMU
  only when subscribed.
* **Correct timestamps.** Odometry and IMU are stamped with the STM32's own
  10 ms sample clock (`stm32_time_correction`). The Jetson's serial thread
  reads the port in ~85 ms bursts, which had turned a 100 Hz signal into
  12 Hz steps.
* **Goals start on the first try.** Because telemetry arrives in bursts, the
  newest odom transform was always 6–60 ms old. Nav2's rotation shim looks
  the robot up at "now" without waiting, and aborted the goal with
  "extrapolation into the future". The odom TF is therefore dated 150 ms
  ahead with a velocity-predicted pose (`tf_future_dating`, as AMCL does for
  map → odom). `default_server_timeout` is 200 ms (50 ms timed out on the
  Nano).

Then:

* `config/nav2_edge.yaml`: DWB only, 10 Hz, 200 trajectories, no debug
  publishing, a map-sized global costmap, lighter AMCL;
* `config/slam_edge.yaml`.

Each change is commented in place.

## Setup / update

**A new Jetson** (JetPack 4.6, a data SD card mounted at `/mnt/sdcard`):

1. The ROS 2 Humble chroot - once, about an hour on the Nano:
   ```bash
   scp src/perceptron_edge/host/setup_chroot.sh jetson@<ip>:
   ssh -t jetson@<ip> bash setup_chroot.sh
   ```
2. The serial bridge, on the host (it needs the host's Python 3 with
   `pyserial`, `pyzmq`, `msgpack` and JetPack's OpenCV):
   ```bash
   scp jetson/jetson_robot_bridge.py jetson@<ip>:
   ```
3. The workspace, maps and robot data, then build and install `robot`:
   ```bash
   rsync -a --exclude __pycache__ src/{perceptron_hardware,perceptron_navigation,perceptron_robot_description,perceptron_robot_control,perceptron_robot_bringup,ct6b_teleop,perceptron_edge} jetson@<ip>:/mnt/sdcard/ros2_chroot/opt/perceptron_ws/src/
   ssh -t jetson@<ip> sudo /mnt/sdcard/ros2_chroot/opt/perceptron_ws/src/perceptron_edge/host/install.sh
   rsync -a src/perceptron_navigation/maps/ jetson@<ip>:/mnt/sdcard/ros2_chroot/opt/perceptron_ws/maps/
   scp config/marker_map.yaml config/dock_station.yaml jetson@<ip>:/mnt/sdcard/ros2_chroot/opt/perceptron_ws/config/
   ```
   `dock_station.yaml` is this robot's taught dock; on another robot or dock,
   leave it out and run `robot dock-teach` instead.

**After changing code**: the `rsync ... src/` line again, then `robot build`
(or `install.sh` when a package's `setup.py` changed) and `robot restart`.
Python and YAML are symlink-installed, so for those a `robot restart` is enough.
