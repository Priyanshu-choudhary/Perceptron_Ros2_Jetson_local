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
