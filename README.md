# Perceptron

A four-wheel ROS 2 robot that maps a room, finds itself in it, drives to goals
and plugs itself into its charger, all on its own Jetson Nano. A laptop is only
needed to watch, through Foxglove. The same workspace also runs the robot in
Gazebo.

- **Mapping**: slam_toolbox, driven by Foxglove goals, the RC transmitter or teleop.
- **Navigation**: AMCL + Nav2 (Smac2D planner, DWB behind a rotation shim),
  tuned on the Nano to about one of its four cores.
- **Charger docking**: `robot dock X Y YAW`. Nav2 brings the robot in front
  of the charger, then the camera and an ArUco board do the last 35 cm to the
  ±3 cm contacts, retrying until the battery current shows it is charging.
- **Goals from anywhere**: Foxglove's goal tool or `robot goal`, including from
  the charger, which the robot backs off first.
- **Optional**: EKF (wheels + IMU), ArUco pose seeding, FlySky CT6B RC control.

## Hardware

| | |
|---|---|
| Computer | Jetson Nano 4 GB, JetPack 4.6 (Ubuntu 18.04), data SD card at `/mnt/sdcard` |
| Base | Four-wheel skid steer, 0.415 × 0.45 m. STM32 motor/IMU board on USB serial: wheel odometry, IMU, battery voltage and current |
| Lidar | LDROBOT STL-19P (LD19 protocol), 360°, 10 Hz, USB serial ([HARDWARE_LIDAR.md](HARDWARE_LIDAR.md)) |
| Camera | USB, 1280×720 MJPEG, ~115° wide, mounted on the lidar. Used only for ArUco markers |
| Battery | 3S LiPo, charged through contacts on the robot's nose |
| Dock | Charger against a wall, a printed 2×2 ArUco board above it (DICT_6X6_250, ids 0-3, 60 mm tiles) |

## How it fits together

```
 STM32 board ──serial─┐
 LD19 lidar ──serial──┤   jetson_robot_bridge.py        host (Ubuntu 18.04)
 USB camera ──────────┘   serial -> ZMQ PUB :5555
                          ZMQ PULL :5556 -> motors
                          camera opened only while docking asks for it
                                     │ ZMQ, 127.0.0.1
 ┌───────────────────────────────────▼─────────────────────────────────────┐
 │ ROS 2 Humble in an Ubuntu 22.04 chroot on the SD card                   │
 │                                                                         │
 │  jetson_bridge_node   /scan /odom /imu /battery   <- /cmd_vel           │
 │  slam_toolbox  or  map_server + AMCL                                    │
 │  Nav2: planner, controller, behaviours, BT navigator ("leave the        │
 │        charger first" step -> dock_guard)                               │
 │  dock (runs only for robot dock/undock/teach/watch)                     │
 │  robot_state_publisher, battery_node, foxglove_bridge                   │
 └───────────────────────────────────┬─────────────────────────────────────┘
                                     │ WebSocket :8765 (the only way in)
                                     ▼
                              laptop: Foxglove
```

- **Why a chroot:** JetPack 4.6 is Ubuntu 18.04 and ROS 2 Humble needs 22.04. The chroot
  is a complete 22.04 on the SD card; the Jetson's own system is untouched.
- **Why the bridge stays outside ROS:** the serial and camera bridge
  (`jetson/jetson_robot_bridge.py`) stays on the host and speaks ZMQ, so it
  works the same whether ROS runs on the Jetson or on a laptop.
- **What `robot` does:** it's a bash command on the host. It mounts the chroot,
  keeps the bridge running, and starts ROS inside the chroot as the
  unprivileged `jetson` user.

## Using it

From an SSH session on the Jetson. `robot` asks for the sudo password once.

```bash
sudo ./robot start slam                  # build a map
sudo ./robot save-map room               # save it
sudo ./robot start nav map=room          # navigate on it
sudo ./robot start gps                   # outdoor GPS navigation + LiDAR avoidance
sudo ./robot waypoint 28.6139 77.2090    # drive to GPS coordinate (lat lon)
sudo ./robot route config/sample_gps_route.yaml # follow GPS waypoint sequence
sudo ./robot pose -0.015 0.145 0         # where the robot is (or Foxglove: Publish pose estimate)
sudo ./robot goal 1.0 0.5 0              # drive there; prints progress, Ctrl-C cancels
sudo ./robot dock -0.341 0.0477 0        # go and charge
sudo ./robot undock                      # back off the charger
sudo ./robot status                      # what is running, battery
sudo ./robot cpu                         # CPU and memory of every part
sudo ./robot logs | stop | restart | help
```

Options for `start`, as `key=value`:

| Option | Effect |
|---|---|
| `map=NAME` | Map to navigate on (nav mode) |
| `ekf=true` | EKF fusion of wheels and IMU |
| `aruco=true` | Seed AMCL from the wall ArUco board |
| `teleop=true teleop_port=/dev/ttyUSB2` | FlySky CT6B receiver |
| `motors=false` | Bench mode: everything runs, the wheels don't turn |
| `foxglove=false` | No Foxglove bridge |
| `log_level=info` | More Nav2 / SLAM logging |

`restart` replays the last options.

**Watching from the laptop**: open Foxglove (the desktop app or
app.foxglove.dev) → *Open connection* → *Foxglove WebSocket* →
`ws://<jetson-ip>:8765`. The 3D panel takes `/map`, `/viz/scan`, `/tf`, both
costmaps, `/plan` and `/local_plan`. Its toolbar publishes pose estimates and
goals. `/dock/status` shows docking progress.

## Charging dock

Teach once: put the robot on the charger so it charges, and run
`robot dock-teach`. It takes about 30 s:
1. It records where the board is relative to the robot.
2. It reverses 25 cm in a straight line to measure the camera's real tilt.
3. It drives back on.

After that, run `robot dock X Y YAW` from anywhere on the map, where X Y YAW is a
spot about 35 cm in front of the charger. It goes through these steps:
1. Nav2 drives there with the camera off.
2. The camera switches on. The robot lines up 22 cm out, drives in, and
   checks its line again 10 cm out.
3. It stops when the battery current turns negative, meaning it's charging.
4. If there's no charge, it pushes 1 cm, wiggles, then backs off and tries again
   aiming slightly left or right, up to 5 attempts.

Measured: from 1.2 m away facing the wrong way, docked on the first attempt in
45 s.

`robot dock` alone reuses the last X Y YAW. `robot dock here` skips Nav2.
`robot dock-watch` shows the live offset from the docked pose and the current
without moving. How it works and every tuning value:
[src/perceptron_edge/README.md](src/perceptron_edge/README.md).

## Where things live on the Jetson

| What | Path |
|---|---|
| ROS 2 chroot | `/mnt/sdcard/ros2_chroot/` |
| ROS 2 Humble | `/mnt/sdcard/ros2_chroot/opt/ros/humble/` |
| Workspace (`/opt/perceptron_ws` inside the chroot) | `/mnt/sdcard/ros2_chroot/opt/perceptron_ws/` |
| Maps | `…/perceptron_ws/maps/`, linked as `~/robot_maps` |
| Taught dock, wall marker map | `…/perceptron_ws/config/` |
| `robot` command | `/mnt/sdcard/perceptron/bin/robot`, linked as `~/robot` |
| Logs (`latest.log` = current run) | `/mnt/sdcard/perceptron/logs/` |
| Serial / camera bridge (outside the chroot) | `/home/jetson/jetson_robot_bridge.py` |

`sudo ./robot shell` opens a shell inside the chroot with ROS sourced, for
`ros2 topic list` and the like.

## Development & Deployment Workflow (Laptop WSL -> Jetson)

> **Note:** The Git repository and primary development environment live on your **laptop (WSL)** (`/home/laptop/perceptron_test_ws`), where there are ample CPU and RAM resources. The Jetson Nano is the runtime target running code in its SD card chroot (`/mnt/sdcard/ros2_chroot/opt/perceptron_ws`).

### Daily Workflow

1. **Develop, test, and commit on the Laptop (WSL):**
   ```bash
   cd ~/perceptron_test_ws
   colcon build --symlink-install
   ```

2. **Send updated code to the Jetson:**
   Sync the ROS 2 packages to the Jetson's SD card chroot over the network:
   ```bash
   # Sync ROS 2 packages
   rsync -avz --exclude '__pycache__' --exclude 'build' --exclude 'install' --exclude 'log' \
     src/{perceptron_hardware,perceptron_navigation,perceptron_robot_description,perceptron_robot_control,perceptron_robot_bringup,ct6b_teleop,perceptron_edge} \
     jetson@<jetson-ip>:/mnt/sdcard/ros2_chroot/opt/perceptron_ws/src/

   # If maps or config were updated:
   rsync -avz src/perceptron_navigation/maps/ jetson@<jetson-ip>:/mnt/sdcard/ros2_chroot/opt/perceptron_ws/maps/
   scp config/marker_map.yaml config/dock_station.yaml jetson@<jetson-ip>:/mnt/sdcard/ros2_chroot/opt/perceptron_ws/config/

   # If the host-side bridge was updated:
   scp jetson/jetson_robot_bridge.py jetson@<jetson-ip>:/home/jetson/
   ```
   *(Replace `<jetson-ip>` with `jetson-desktop.local` or the Tailscale IP `100.80.225.114`).*

3. **Apply and restart on the Jetson:**
   ```bash
   ssh jetson@<jetson-ip>

   # For Python/launch/config changes (symlink-installed):
   sudo ./robot restart

   # If package structure, setup.py, or C++ dependencies changed:
   sudo ./robot build && sudo ./robot restart
   ```

## Setting up a Jetson

In short:
1. `setup_chroot.sh` builds the ROS 2 chroot (once, about an hour).
2. Copy the bridge to `/home/jetson`.
3. rsync `src/` into the chroot's workspace.
4. `install.sh` builds it and installs `robot`.
5. Copy the maps and `config/` files.

The exact commands, and how to update after changing code, are in
[src/perceptron_edge/README.md → Setup / update](src/perceptron_edge/README.md#setup--update).

## Repository layout

| Path | What it is |
|---|---|
| [`src/perceptron_edge`](src/perceptron_edge/README.md) | The Jetson setup: launch file, Nano-tuned Nav2 config, `robot` CLI, docking, `dock_guard`, Foxglove config |
| [`src/perceptron_hardware`](src/perceptron_hardware/README.md) | `jetson_bridge_node` (ZMQ → ROS), `battery_node`, STM32 serial bridge |
| [`src/perceptron_navigation`](src/perceptron_navigation/README.md) | Nav2 / SLAM / AMCL parameters, behaviour trees, maps, ArUco localiser, missions, path overlay |
| [`src/perceptron_robot_description`](src/perceptron_robot_description/README.md) | URDF / xacro, meshes |
| [`src/perceptron_robot_bringup`](src/perceptron_robot_bringup/README.md) | Laptop launch files: Gazebo, real robot over the bridge, localisation, EKF |
| [`src/perceptron_robot_control`](src/perceptron_robot_control/README.md) | `ros2_control` drivetrain configuration (the URDF needs it, so it is built on the Jetson too) |
| [`src/perceptron_robot_gazebo`](src/perceptron_robot_gazebo/README.md) | Gazebo world with the simulated ArUco docking station |
| [`src/perceptron_docking`](src/perceptron_docking/README.md) | Earlier docking for the simulated camera (image topics); the real robot uses `perceptron_edge` |
| [`src/aruco_detection`](src/aruco_detection/README.md) | Legacy, not built (`COLCON_IGNORE`) |
| [`src/ct6b_teleop`](src/ct6b_teleop/README.md) | FlySky CT6B RC transmitter → `/cmd_vel` |
| [`jetson/`](jetson/README.md) | Code that runs on the Jetson host, outside ROS: the serial / camera bridge, the AR path overlay |
| [`tools/perf/`](tools/perf/README.md) | The CPU / memory measurements behind the Nano tuning |
| `tools/` | Gyro calibration and test scripts, udev rules |
| `config/` | This robot's data: wall marker map, taught dock station |

## Performance on the Nano

Whole board, all four cores, MAXN mode. Includes the serial bridge and the
desktop.

| State | Cores busy (of 4) | Memory |
|---|---|---|
| nav, localised, idle | ~1.0 | ~360 MB |
| nav, driving to a goal | ~1.05 | ~400 MB |
| slam + Nav2 | ~1.05 | ~360 MB |
| + Foxglove streaming to a laptop | +0.1 | +20 MB |
| docking (camera detection at 10 Hz) | +0.4 while it runs | |

The untuned laptop configuration needed 2.7 cores and missed 299 control deadlines
in 30 s on the Nano. What changed and why, measured one switch at a time:
[src/perceptron_edge/README.md → What was optimised](src/perceptron_edge/README.md)
and [tools/perf/](tools/perf/README.md).

## Running ROS on a laptop instead

**Simulation**: build the workspace (restore the two vendored packages first,
see [VENDOR_PACKAGES.md](VENDOR_PACKAGES.md)), then

```bash
ros2 launch perceptron_robot_bringup gazebo_control2.launch.py
ros2 launch perceptron_navigation nav_simulation.launch.py slam:=true
```

**The real robot with ROS on the laptop**: the Jetson runs only the serial
bridge, and the laptop runs the rest.

```bash
sudo ./robot bridge                         # on the Jetson (ROS stopped there)
ros2 launch perceptron_robot_bringup nav.launch.py jetson:=<jetson-ip> map:=<map.yaml> ekf:=false
```

## Known issues

- **Charger contacts**: once, the whole robot lost power the instant it
  pressed onto the contacts (battery at 11.6 V). Check that no pad can bridge
  `+` and `−` when the nose arrives off-centre, and fix the charger box to the
  floor. A robot that misses pushes it along.
- **STM32 USB link**: it sometimes stalls under motor load, more often on a low battery.
  The bridge reopens it in ~5 s, and docking stops and waits.
- **Camera tilt**: the camera points 5.2° lower than the URDF says. Docking
  measures and corrects this itself, but the ArUco localiser and the path
  overlay use the URDF.
- **Duplicate marker ids**: the dock board and the `wall_north` board in
  `config/marker_map.yaml` share ids 0-3. Re-survey the marker map before
  using `aruco=true` if the wall board was moved to the dock.
- **DDS shared memory is off**: ROS on the Jetson uses Fast DDS over UDP on
  127.0.0.1 (`src/perceptron_edge/config/fastdds_localhost_udp.xml`). Shared
  memory got stuck after short-lived ROS commands and held the idle Nano at
  3.8 of 4 cores. `robot` and `robot shell` set this up. A shell opened some
  other way needs `FASTRTPS_DEFAULT_PROFILES_FILE` and `ROS_LOCALHOST_ONLY=0`.
