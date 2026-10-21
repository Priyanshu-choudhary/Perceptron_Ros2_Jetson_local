# perceptron_perception

Camera + 3D lidar perception for the Perceptron rover. Every node takes plain ROS topics
(`PointCloud2`, `Image`, `CameraInfo`, TF), so the same code runs on the CARLA bridge and on the
robot. The geometry lives in ROS-free modules with unit tests; the `*_eval` / `*_check` nodes
score against CARLA ground truth and are never part of the robot pipeline.

| Node | Roadmap | What it does |
|---|---|---|
| `lidar_camera_overlay_node` | Master plan step 2 | LiDAR points drawn on the camera image, coloured by depth |
| `projection_check_node` | step 2 / task 4.1 | LiDAR→camera alignment in pixels per range bin |
| `lidar_preprocessor_node` | Stage 4 | range + self-box filter, ground removal, clusters, `/scan` |
| `cloud_eval_node` | Stage 4 | scores the preprocessor against CARLA's semantic LiDAR |
| `object_fusion_node` | Stage 5 | 2D boxes + LiDAR clusters → objects with class, 3D position, range |
| `fusion_eval_node` | Stage 5 | range error, association correctness and coverage vs ground truth |

## Stage 4: LiDAR preprocessing

```bash
ros2 run perceptron_carla carla_bridge --crossing 6                 # people pacing across the road
ros2 run perceptron_carla carla_bridge --autopilot --traffic 30 --pedestrians 40
ros2 launch perceptron_perception carla_lidar.launch.py             # preprocessor + scoring (eval:=false to skip)
rviz2 -d $(ros2 pkg prefix perceptron_carla)/share/perceptron_carla/rviz/carla.rviz
```

```
/points (lidar3d_link) --range 0.5-50 m--> base_footprint --self box--> voxel 8 cm
   --> ground: RANSAC plane (5 cm fit) + local-height grid   -> /perception/points_ground
   --> obstacles below 2.5 m, 2D Euclidean clusters (0.5 m)  -> /perception/points_obstacles (x,y,z,cluster)
                                                                 /perception/clusters (Detection3DArray)
                                                                 /perception/cluster_markers
                                                                 /perception/scan (LaserScan, 0.1-1.5 m band)
                                                                 /perception/lidar_preprocess_stats
```

Design notes are in `cloud_processing.py`:
- **Plane plus grid.** One plane calls a 15 cm sidewalk an obstacle; a pure grid calls a car's underside ground.
- **Tight RANSAC threshold.** At 10 cm RANSAC bridges road and sidewalk with a 1° tilted plane.
- **2D clusters.** A 16-beam LiDAR's rings are 0.35-0.7 m apart vertically, which cuts a person
  into slices in 3D.

Cluster ids are per frame; persistent track ids are Stage 6.

### Results (CARLA, rover mount, 16-beam lidar at 0.415 m pitched 15° down)

| Task | Measure | Result |
|---|---|---|
| 4.1 frames | fitted ground plane in base_footprint | tilt **0.33°**, height **+0.003 m**, so the extrinsic is right; also ≤0.4 px camera alignment |
| 4.2 self / range | host Microlino points | **40 %** of the raw cloud → **0.00 %** after the self box |
| 4.3 ground | precision / recall | **0.96-0.97 / 0.994-0.998**; misses are low props, fence and building bases, feet |
| 4.4 clusters | pedestrians 0-10 / 10-20 / 20-40 m (`--crossing 6`, 12.6 k actor-frames) | **100 / 100 / 99.6 %** matched, 0 % merged |
| | vehicles 10-20 / 20-40 m (city traffic) | **95 / 76 %** matched; 41 % of close cars split in 2 |
| | cones and boxes 0-10 / 10-20 / 20-40 m | 100 / 100 / 50-99 % |
| 4.5 scan | `/perception/scan` | 720 bins, nearest obstacle in the 0.1-1.5 m band |
| 4.6 latency | processing per cloud (about 14.5 k points in) | **11-15 ms** mean, p95 ≤ 16 ms, at 20 Hz input |

**What limits it is the sensor geometry, not the processing.** In the URDF the 3D lidar is pitched 15°
down with a ±15° field of view, so its **top beam is horizontal at 0.42 m**: nothing above knee
height is seen, and a person 10-40 m away collects only 4-7 points. In 50 s of city driving with
80 pedestrians on the sidewalks, the lidar hit a person in 10 of 968 frames. For detecting
people and cars, a smaller pitch (or a wider vertical FOV) would help a lot. The camera has to
carry most of the object detection, and that's exactly what Stage 5's camera-lidar
association is for. `cluster_min_points` is 3 in `config/carla.yaml` (default 6) because of these sparse
returns. The cost is about 10 more small scenery clusters per frame.

Close cars split in two (41 % at 10-20 m): the lidar sees a car's front and side as separate
faces with a gap at the wheel arch. Stage 5 merges clusters that fall inside the same camera box.

### Scoring (`cloud_eval_node`)

CARLA's semantic lidar fires the same rays from the same pose and labels every return with its
class and actor id. Each processed point takes the label of its nearest twin (≤ 10 cm). Then:
ground precision/recall; host-vehicle share before and after the self box; per actor with ≥ 3
obstacle points: matched (one cluster holds ≥ 50 % of it and is ≥ 50 % it), split, merged; plus
the preprocessor's timing and fitted plane.

## Stage 5: camera–LiDAR association

```bash
ros2 run perceptron_carla carla_bridge --crossing 6          # or --scenario, or --autopilot --traffic 60
ros2 launch perceptron_perception carla_fusion.launch.py      # Stage 4 + fusion + scoring
ros2 run rqt_image_view rqt_image_view /perception/objects_image
```

`object_fusion_node` takes any `vision_msgs/Detection2DArray`. In CARLA that's the reference
detector `/carla/ground_truth/boxes_2d`; set `detections_topic` to swap in your own model. It
publishes, stamped with the source frame:

| Topic | Content |
|---|---|
| `/perception/objects` | `Detection3DArray` in base_footprint: id and class from the 2D box, bbox = extent of its points, hypothesis pose = centroid **with covariance** |
| `/perception/objects_json` | every box: id, class, score, **range** (m from the camera to the nearest surface; `null` without LiDAR), x/y/z, points, how it was associated |
| `/perception/object_points` | the points used per object (field `object` = index in the JSON list) |
| `/perception/object_markers`, `/perception/objects_image` | RViz markers; camera image with box, class, range and (x, y) |

**Association** (`object_fusion.py`) works per Stage 4 cluster, not per point:
- A box frustum also holds the wall behind the object, and a pedestrian in front of a car sits inside the car's box.
- Clusters are scored by *horizontal* overlap × share inside the box. Horizontal only, because the LiDAR sees only the bottom 0.42 m of things.
- A second pass re-joins split faces of one car: a cluster that sits inside exactly one box joins it if it's within 2.5 m depth of that box's main cluster.
- Boxes without a cluster fall back to the nearest group of loose points.

**Range is to the nearest surface** (5th percentile of point distances from the camera).
The LiDAR sees only an object's near side, so its centroid sits up to half the object's depth in
front of the real centre: about 2–3 m for cars (see the centroid column below).

### Results (reference detector = CARLA visible boxes; error vs the true closest point of each 3D box)

| Case | True range | With a range | Correct points | Range error bias / MAE / p95 |
|---|---|---|---|---|
| **pedestrians, moving** (`--crossing 6`) | 0-10 m | 99.6 % | 99.6 % | +0.03 / **0.10** / 0.15 m |
| | 10-20 m | 94.9 % | 98.9 % | −0.02 / **0.14** / 0.15 m |
| | 20-40 m | 54.6 % (8 % not hit) | 97.3 % | −0.20 / 0.25 / 0.18 m |
| **cones and boxes, static** (`--scenario`) | 0-10 / 10-20 m | 100 / 100 % | 100 % | MAE **0.14 / 0.12** m |
| | 20-40 m | 50 % (50 % not hit) | 100 % | MAE 0.22 m |
| **cars, moving** (city, 60 cars) | 0-10 m | 99-100 % | 100 % | MAE **0.07-0.15** m |
| | 10-20 m | 75-100 % | 93-100 % | MAE 0.2-0.6 m |
| | 20-40 m | 48-100 % | 24-100 % | MAE 0.3-1.9 m, p95 up to 11 m |

Ranges are per 10 s period; the spread shows how much depends on what is in view. Latency is
5–7 ms per frame on top of Stage 4's 11–15 ms. Centroid vs box centre is 0.2–0.4 m for people
and cones, and 2–3 m for cars, almost all along the viewing direction.

**Failure cases**
1. **People beyond ~20 m usually get one LiDAR point or none.** Only the horizontal top beam
   reaches them (sensor geometry, Stage 4). One point is never trusted, so there's no range;
   the camera box still exists. The evaluator reports these as "not hit".
2. **Occluded cars at 20-40 m sometimes take the wrong cluster.** A partly hidden car's
   visible box overlaps the car in front of it, and the bad periods show 24 % correct points and
   8-11 m errors. A depth sanity check from the box height (known class size) is the next fix
   if this matters for the costmap.
3. **The centroid isn't the object centre** (near-face bias). Stage 6 should track the
   nearest-surface point or add half the class depth along the viewing ray, not trust the centroid.

## Stage 6: multi-object tracking

```bash
ros2 run perceptron_carla carla_bridge --crossing 6                             # people, known speeds
ros2 run perceptron_carla carla_bridge --scenario --walkers 0 --ego-speed 1.0   # robot moves, objects static
ros2 run perceptron_carla carla_bridge --autopilot --traffic 60 --pedestrians 40 --seed 3
ros2 launch perceptron_perception carla_tracking.launch.py                      # Stages 4-6 + scoring
# on the Stage 3 EKF instead of perfect odometry: start this first, then the bridge with
#   --tf-mode ekf --odom-vy-cov 0.001
ros2 launch perceptron_carla sim_localization.launch.py fixes:=true
```

`object_tracker_node` → **`/perception/tracked_objects`** (`perceptron_msgs/TrackedObjectArray`,
in `odom`): `id` (stable, generated by the tracker), `class_id` + `class_confidence`,
`pose` and **`twist`** (vx, vy in m/s), both with covariance, `size`, `age`, `hits`, `misses`, `state`
(TENTATIVE / CONFIRMED / COASTING), `moving`. Also `/perception/track_markers` (velocity arrows
showing the next second). Plan around CONFIRMED and COASTING tracks only.

How it works (`tracker.py`):
- **Ego-motion compensation.** Detections are moved into `odom` through TF at their own stamp, so a
  parked car keeps its odom position while the robot drives.
- **Filter.** Per-track constant-velocity Kalman filter on x, y, vx, vy.
- **Association.** Mahalanobis gate + Hungarian, same class only.
- **Lifecycle.** Tentative → confirmed after 3 hits; coasting up to 1 s.
- **Manoeuvre handling.** Established tracks win ties; a missed track re-acquires a detection near its last
  measured position and resets its velocity; duplicate tracks are pruned.
- **Near-face correction.** Detections are pushed back by half the class depth along the ray (Stage 5 bias).

The node never blocks on TF. In ekf mode the EKF's transform arrives a tick after the
detections, and waiting in a callback starves the node's own `/tf` subscription (single-threaded
executor). Messages are queued and processed once their transform exists.

`tracking_eval_node` matches confirmed and coasting tracks to true actors by position (Hungarian,
1 m for people, 2.5 m for cars), not by the detector's label. `carla_bridge` publishes the truth with
real velocities on `/carla/ground_truth/objects`.

### Results

| Scenario | Odometry | Class | ID switches | Position MAE | Velocity error MAE: moving (p95) / static | Speed, heading |
|---|---|---|---|---|---|---|
| `--crossing 6`, 9 082 actor-frames | gt | pedestrian | **8** (0 swaps) | 0.09 m | **0.15** (0.55) m/s / — ¹ | 0.10 m/s, 4.9° |
| `--scenario --ego-speed 1.0` (robot moving) | gt | cones, boxes | **0** | 0.14 m | — / **0.02** m/s | — |
| city, 60 cars, robot on autopilot | gt | vehicle | 20 over 3 490 | 1.16 m | 1.00 (3.4) m/s / 0.53 m/s | 0.75 m/s, 4.6° |
| same route | **Stage 3 EKF** | vehicle | 51 over 3 570 | 1.55 m | 1.19 (3.4) m/s / 0.64 m/s | 0.85 m/s, 6.1° |

¹ The crossing pedestrians reverse instantly at the end of their path, which is unphysical. Their
"static" frames are those one-tick reversals, not standing people; the moving-ego row is the real
static test.

**Status against the plan's target (speed error < 0.5 m/s):**
- **Pedestrians and static objects meet it.**
- **Cars don't yet.** Speed error is about 0.8 m/s at typical 6-10 m/s. Lidar only sees a car's near
  faces, so the point centroid slides along the car as the viewing angle changes, and the filter reads
  that as motion. Stronger smoothing (`vehicle.sigma_a 1.5`, `r_floor 0.6`) helped only a little.
  The fix is **L-shape box fitting**: once a track's heading is known from its velocity, fit a class-sized
  box to the visible faces and track the box centre.
- **On the EKF**, heading and yaw-rate error rotate the scene. A 0.01 rad/s yaw-rate error makes a car
  20 m away appear to move 0.2 m/s. That accounts for most of the gt → EKF difference.

## Stage 7: trajectory prediction

```bash
ros2 run perceptron_carla carla_bridge --crossing 6        # or --autopilot --traffic 60 --pedestrians 40 --seed 3
ros2 launch perceptron_perception carla_prediction.launch.py   # Stages 4-7 + scoring against CARLA's future
```

`trajectory_predictor_node` → **`/perception/predicted_objects`**
(`perceptron_msgs/PredictedObjectArray`, odom, stamped with the tracks' time). It holds, per
confirmed or coasting track:
- the model used (STATIC / CV / CTRV), with the speed and yaw rate it used
- footprint `radius`
- `poses` at **0.5 / 1 / 2 s**, each with 2D covariance
- a dense `path` every 0.1 s

This is the Stage 8 costmap input. `/perception/prediction_markers` draws the path and 2-sigma
ellipses (plus the footprint) in RViz. The tracker message gained `state_covariance` (full
[x, y, vx, vy] covariance) because pose and twist covariance leave out the position-velocity
cross terms that the uncertainty propagation needs.

Each real-world problem and how it is handled (`prediction.py`):

| Problem | Handling |
|---|---|
| **Error amplification** (±0.3 m/s → ±0.6 m at 2 s) | velocity shrunk by its own uncertainty, v·\|v\|²/(\|v\|²+σᵥ²), so a velocity no bigger than its noise is pulled to zero |
| **Static vs dynamic** | asymmetric hysteresis on the tracker's `moving` flag: dynamic after 1 frame (the tracker already requires speed > 2σ), static again only after 10 quiet frames, so a parked car never creeps |
| **Uncertainty expansion** | full covariance propagated through the motion model plus white-acceleration process noise, plus a per-class **systematic position floor**. The tracked centre carries a near-face bias the tracker's random-noise covariance can't know about (0.15 m people and props, **1.2 m cars**, measured) |
| **Physical limits** | per-class speed caps (person 3, car 20 m/s); CTRV turn rate ≤ min(v/R_min, a_lat_max/v); each ellipse capped to the distance the object could physically reach in t |
| **Turning vehicles** | CTRV from a least-squares heading-rate fit. **Off by default** (`use_ctrv: false`): measured *worse* than CV on CARLA traffic (5.2 vs 4.1 m at 2 s), because car headings from the near-face centroid are too noisy. Turn it on once L-shape fitting gives clean headings |

### Results: error against where CARLA actors really were 0.5 / 1 / 2 s later

`prediction_eval_node` computes two baselines from the same track state:
- **stay**: assume the object stays put
- **raw CV**: raw velocity, no gate, shrinkage or caps

| Case | Horizon | **Ours** MAE (p95) | Stay | Raw CV | In 95 % ellipse |
|---|---|---|---|---|---|
| walkers, moving (`--crossing 6`, ~8.7 k per horizon) | 0.5 s | **0.18** (0.59) m | 0.48 | 0.17 | **94.8 %** |
| | 1.0 s | **0.35** (1.27) m | 0.95 | 0.32 | 91.7 % |
| | 2.0 s | **0.86** (3.02) m | 1.80 | 0.79 | 90.3 % |
| cars, moving (city, 60 cars, ~1.6 k) | 0.5 s | **1.64** (3.59) m | 4.30 | 1.55 | 92.5 % |
| | 1.0 s | **2.26** (6.03) m | 7.67 | 2.07 | 87.2 % |
| | 2.0 s | **3.86** (13.5) m | 14.18 | 3.50 | 87.5 % |
| cars, stopped | 0.5 / 1 / 2 s | **1.41 / 1.63 / 2.20** m | 1.23 / 1.26 / 1.41 | 1.50 / 1.84 / 2.65 | **98.6 / 97.5 / 94.8 %** |
| static props, robot moving at 1 m/s | 0.5-2 s | **0.14** m, flat | 0.14 | 0.14-0.15 | (floor added since) |

What the numbers say:
- **Against "stay"**: 2.7× better for walkers and 3.7× better for moving cars at 2 s.
- **Against raw CV**: within 5-10 % on moving objects, and **17 % better on stopped cars**, where gating stops them creeping. That trade is the right one for a costmap: phantom motion of parked cars blocks free space all the time, while the moving-object cost is mostly cars pulling away from a stop.
- **Calibration**: the 95 % ellipses really hold 87-99 % of true positions. That's honest enough for Stage 8 to inflate costs by them; it was 59-89 % before the systematic position floor was added.
- **Car error starts at about 1.2 m at every horizon.** That's the Stage 6 centroid offset ("stay" on stopped cars). The same L-shape box fitting that would fix Stage 6's car velocity is the biggest single improvement left here too, and it would make CTRV usable.

The crossing walkers reverse instantly at the end of their path. Their "truly static" frames are
those one-tick reversals, and the 10 % of moving walkers predicted static are tracks right after a
reversal or at start-up. Real pedestrians decelerate, so this scenario is a pessimistic case.

## Stage 8: rolling 3D-aware costmap with a dynamic risk layer

```bash
ros2 run perceptron_carla carla_bridge --autopilot --traffic 40 --pedestrians 80 --no-semantic
ros2 launch perceptron_perception carla_costmap.launch.py        # Stages 4-8 + scoring
rviz2 -d $(ros2 pkg prefix perceptron_carla)/share/perceptron_carla/rviz/carla.rviz
```

`costmap_node` keeps a 40 × 40 m grid in `odom` (0.2 m cells, 5 cm voxel height), centred on
the robot and shifted by whole cells. It publishes, each stamped with the LiDAR frame it includes:

| Topic | Layer |
|---|---|
| `/costmap/voxels` (PointCloud2) | **1: 3D voxels.** Occupied voxel centres; z = height above the ground |
| `/costmap/obstacles` (OccupancyGrid) | 1 projected to 2D over the robot's height band: 100 / 0 / −1 unknown |
| `/costmap/inflated` | **2: inflation.** Nav2 convention: lethal, inscribed, exponential fall-off |
| `/costmap/dynamic_risk` | **3: dynamic risk.** Predicted paths, weighted by time-to-encounter |
| `/costmap/combined` | max(2, 3): the planner input for Stage 11 |
| `/costmap/stats` | [ms, voxels, lethal cells, risk cells, moving objects, data age] |

How it works (`costmap.py`):
- **Voxel log-odds.** Time decay bounds persistence. Rays through a voxel clear it, and so does ground
  seen in a column with no obstacle in it: from a 0.42 m LiDAR, rays to the road arrive too flat to
  clear the space a car has just left. Moving objects' points are kept out of the voxel layer, so a
  walking person leaves no trail.
- **Height band.** 0.05 m to `max_obstacle_height`: bumps and overhangs are not obstacles.
- **Dynamic risk.** Gaussians along each predicted path with the Stage 7 covariance at that moment,
  widened by object + robot radius, weighted by max(near-term, robot-arrives-at-the-same-time).
  The current footprint is lethal.

The footprint in `config/carla.yaml` is the **Microlino host** that drives in CARLA (inscribed
0.75 m, inflation 1.8 m). The node defaults are the rover's own Nav2 values (0.21 / 0.6 m,
scaling 2.5), so on the robot only the input topics change.

### Results (`costmap_eval_node` vs CARLA ground truth)

| | city: 40 cars, 80 people, robot on autopilot | `--crossing 6` (people crossing ahead) |
|---|---|---|
| obstacle recall (true footprint under a lethal/inscribed cell) | stopped cars **100 %**, moving cars **99.3 %** | walking **98.9 %**, standing 97.6 % |
| false obstacles (lethal where the LiDAR saw only ground) | **2.0 %** (7.7 % before ground-column clearing) | **0.95 %**, all sidewalk edges |
| dynamic risk at the true position 1 s / 2 s later | 50 % / 52 % | **95.0 % / 94.3 %** |
| flicker (cells toggling lethal per update) | 0.03 % | 0.00 % |
| update time mean / p95 | 35 / 49 ms | 49 / 54 ms (6 moving people to paint) |

Notes:
- **Dynamic coverage for cars is about 50 % by design, and limited by prediction error.** Cells the
  robot can't reach when the car gets there are deliberately weighted down, and Stage 7's car
  predictions carry the ~1.2 m centroid offset. For people, whose speed is close to the robot's,
  95 % of true future positions are inside the risk layer.
- **Unknown space is reported as unknown.** In CARLA the host car hides the area behind the LiDAR,
  so the map shows unknown there instead of free.
- **Update time is about the 50 ms tick budget.** About 25-35 ms of it is the voxel ray clearing
  (numpy). If the robot needs headroom: fewer rays (`max_rays`), 0.25 m cells, or move the
  ray loop to C++/numba.

## Road and lane-line map from segmentation (visualisation; input for later planning)

```bash
ros2 run perceptron_carla carla_bridge --autopilot --traffic 40 --pedestrians 80   # semantic camera ON
ros2 launch perceptron_perception carla_costmap.launch.py                           # includes it (road:=false to skip)
```

`road_segmentation_node` turns each road / sidewalk / road-line pixel of `/carla/camera/semantic` into a
3D point and keeps the latest label per 0.1 m cell of a 40 m rolling grid in `odom`, so the road stays
painted after it leaves the camera view. It publishes `/perception/road_cloud` (x, y, z, rgb: green road,
magenta sidewalk, orange markings) at 5 Hz.

| `projection` | 3D from | Use |
|---|---|---|
| `depth` (CARLA default) | the depth image | exact |
| `ground` | pixel ray meets the ground plane z = 0 of `base_footprint` | the real rover: a segmentation model on its camera, no depth camera needed. Markings are on the ground, so they come out right; a raised kerb lands slightly too far |

All marking pixels are used (they are a few pixels wide at range); road and sidewalk pixels are taken
every `stride`-th row and column. The class ids are parameters (`road_ids`, `sidewalk_ids`, `line_ids`,
default CARLA's 1 / 2 / 24), so a Cityscapes-style model needs only those changed.

Measured in CARLA (Town10, autopilot): 9-14 ms per image at 10 Hz, about 45 k road, 8-15 k sidewalk and
4-10 k marking cells. In both modes the camera's markings land on the lines of CARLA's own map
(`/carla/map/lanes`), which RViz draws on top (`samples/road_lanes*.png` on the Windows side).

Limits: CARLA has one `road_line` class, so the map can't tell white from yellow or solid from dashed;
crosswalks and stop lines are also `road_line`. The costmap doesn't use it yet: an off-road cost
layer built from it is a candidate for Stage 11.

## Stage 9 change to Stage 5: size gate for pedestrian boxes

The Stage 9 city scenario showed that sidewalk pedestrians are almost never hit by the rover LiDAR (top
beam level at 0.42 m, the rest pitched down), yet fusion gave 17 % of their boxes a range: the wall or
parked car behind them, 12 m too far on average. `object_fusion.associate` now takes a per-box `max_depth`.
For a pedestrian box h px tall it is `size_gate_factor (1.15) · fy · 1.95 m / h + 0.5 m` (the deepest a
person can be and still look that tall). Clusters and loose points beyond it are refused, so the box
gets no range instead of a wrong one. Results: wrong ranges 17 % → 11 % of boxes, their error 12.4 m →
6.8 m. The crossing scenes are unchanged (99 % correct, MAE 0.15 m). Other classes get no gate. Turn it
off with `size_gate: false`. See perceptron_carla's README, Stage 9.
