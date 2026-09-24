# tools/perf — CPU / memory profiling of the nav stack

Measures what `nav.launch.py` costs, one switch at a time, on the laptop or on
the edge board. Nothing here edits the packages: every variant is a launch
argument, a generated copy of `nav2_params.yaml`, or a patched copy of one node.

The robot stays parked. `profile_nav.launch.py` remaps the bridge's `/cmd_vel`
to a sink, so Nav2 plans and controls for real but the motors never move.

## What a run does

`profile_run.py` starts the launch, seeds AMCL, then measures two windows:

* **idle**: localised, no goal.
* **nav**: `/goal_pose` sent. `controller_server` gets its odometry from a
  loop-back of the velocity Nav2 commanded, so DWB samples as if it were
  cruising. AMCL is asked for N filter updates/s (`--nomotion-hz`) as if the
  robot were moving.

CPU is exact CPU time from `/proc/<pid>/stat` for every process in the launch
tree. Memory is PSS. Output goes to `results/NAME.json`, plus the full launch
log.

## Laptop

```bash
python3 make_params.py base dwb_nodebug ctrl_reduced cm_lean plugins_lean optimized
./run_laptop.sh BASE_01 --launch-arg ekf:=false --params-file params/base.yaml
./run_laptop.sh rviz_off --launch-arg ekf:=false --launch-arg rviz:=false --params-file params/base.yaml
./run_laptop.sh BASE_02 --launch-arg ekf:=false --params-file params/base.yaml
python3 analyze2.py results      # each run vs the baselines either side of it
```

The laptop drifts by up to 20% as it heats up, so always bracket a change with
baselines (A-B-A). `analyze2.py` interpolates the baseline in time.

To stop the robot's battery and Wi-Fi from changing the input between runs:

1. Record once with `replay/record.py 90 stream.pkl`.
2. Run `replay/fake_jetson.py stream.pkl 100.8 38` and use `JETSON=127.0.0.1`.

The last argument keeps only the first 38 s of scans. Cut any stretch where
someone walks past the robot.

Launch switches in `profile_nav.launch.py`:

* the stock arguments;
* `aruco`;
* `bond` (off scopes `bond_timeout: 0.0` to the Nav2 include only);
* `use_composition`;
* `lean_bridge` (runs `lean/jetson_bridge_node_lean.py`, the 4-line patch in
  `lean/*.patch`).

## Edge board (Jetson Nano measured; Pi 5 the same way)

On JetPack 4.x there is no ROS 2 Humble, so `src/perceptron_edge/host/setup_chroot.sh` builds an
Ubuntu 22.04 arm64 chroot on the data SD card with only the Nav2 packages the
stack uses. On a Pi 5 running Ubuntu 22.04, install the same packages natively
and skip the chroot.

Stage these in `/work`:

* `edge_nav.launch.py`
* `profile_run.py`
* the parameter files
* `pylib/perceptron_hardware` and `pylib/perceptron_navigation`
* the BT XMLs, the map and a generated `robot.urdf`

Then run `edge/run_edge.sh NAME ...` inside the chroot.

On a Cortex-A57 (Jetson Nano), load **DWB only**
(`controller_plugins: [FollowPath]`). The Humble arm64 MPPI plugin kills
`controller_server` with SIGILL there.

Remove the chroot:

```bash
for m in dev/shm dev/pts dev sys proc; do sudo umount /mnt/sdcard/ros2_chroot/$m; done
sudo rm -rf /mnt/sdcard/ros2_chroot /mnt/sdcard/ros2_dl
```

Always unmount first. `rm -rf` through a live `/dev` bind mount deletes host
device nodes.

## Benchmarks

`bench/bench.cpp` and `bench/bench.py` are the same kernels on every machine.
They cover costmap raytrace, inflation, DWB rollouts, the AMCL likelihood
field, Python scan conversion and thread wake-ups. Compile with
`g++ -O2 -std=c++14 -pthread bench.cpp`. The Nano ÷ laptop ratio is per kernel
type, because wake-ups and compute scale very differently under WSL2.

## Gotcha found while building this

A global `SetParameter` in a launch file adds `-p` to every node. Any `-p`
makes a node's own YAML section beat its inline dict. AMCL then silently
took `use_sim_time: true` and `set_initial_pose: true` from
`nav2_params.yaml`. Scope `SetParameter` inside a `GroupAction`.
