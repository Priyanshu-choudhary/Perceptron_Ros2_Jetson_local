#!/usr/bin/env python3
"""
Synchronous CARLA -> ROS 2 (Humble) bridge for the Perceptron rover. Runs in WSL;
CARLA runs natively on Windows. See ../README.md for setup, topics, TF ownership.

Everything comes from the SAME simulation tick and carries that tick's sim-time stamp.

Robot-equivalent inputs (same names, frames and message shapes as the real rover)
  /odom                          Odometry   wheel odometry: twist only, noisy  (stm32_bridge_node)
  /imu/data                      Imu        attitude + rates + accel, noisy    (pixhack_bridge_node)
  /gps/fix                       NavSatFix  noisy fix at --gnss-hz             (pixhack_bridge_node)
  /carla/camera/image_raw        Image bgr8, /carla/camera/camera_info
  /carla/lidar/points            PointCloud2 in lidar3d_link
Ground truth (evaluation and training ONLY -- never feed these to the EKF or a model at runtime)
  /carla/ground_truth/ego_odom   Odometry   perfect base_footprint pose in carla_world
  /carla/camera/depth            Image 32FC1 metres
  /carla/camera/semantic         Image mono8 class id (names: /carla/camera/semantic_labels)
  /carla/camera/semantic_color   Image bgr8
  /carla/camera/instance         Image 16UC1 CARLA instance id (NOT actor id)
  /carla/camera/instance_color   Image bgr8
  /carla/ground_truth/boxes_3d        Detection3DArray in carla_world
  /carla/ground_truth/boxes_2d        Detection2DArray, visible part; score = visible fraction
  /carla/ground_truth/boxes_2d_amodal Detection2DArray, projected 3D box
  /carla/camera/boxes_debug      Image bgr8, /carla/ground_truth MarkerArray
  /carla/ground_truth/lidar_semantic  PointCloud2 x,y,z,obj_idx,obj_tag: same rays as the lidar
  /carla/ground_truth/objects   perceptron_msgs/TrackedObjectArray, carla_world: true pose + velocity
  /carla/bridge_info             std_msgs/String (latched) JSON: tf_mode, mount, ego_id, host box
  /carla/map/lanes               MarkerArray (latched), carla_world: lane centres, markings, road edges
Visualisation
  /carla/ego_marker              MarkerArray (latched), base_footprint: the host car + sensor rig
  /clock

TF (each transform has exactly one publisher; see README "TF ownership")
  carla_world -> map             static, this bridge (identity in gt mode, start pose in ekf mode)
  map -> odom                    this bridge in gt mode; ekf_filter_node_global or a static
                                 identity from sim_localization.launch.py in ekf mode
  odom -> base_footprint         this bridge in gt mode; ekf_filter_node_local in ekf mode
  base_footprint -> base_link -> {camera_link -> camera_optical_link, lidar3d_link, imu_link, gps_link}
                                 static, this bridge, values from perceptron_robot.xacro

Detection id = CARLA actor id everywhere (boxes, markers, CSV), so 2D, 3D and tracks line up.
Instance ids are matched to actors by back-projecting a few of each instance's pixels with the
depth image and checking which actor's 3D box they fall in, because CARLA's instance ids are
internal object ids and do not equal actor ids (measured on 0.9.15).

Usage:
  ros2 run perceptron_carla carla_bridge --scenario                                # static test objects
  ros2 run perceptron_carla carla_bridge --traffic 40 --pedestrians 60 --autopilot # busy city
  ros2 run perceptron_carla carla_bridge --autopilot --tf-mode ekf                 # localization test
"""
import argparse
import array
import csv
import json
import math
import os
import queue
import random
import re
import subprocess
import time

import cv2
import numpy as np
import carla

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, CameraInfo, Imu, NavSatFix, NavSatStatus, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import (Detection2D, Detection2DArray, Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from perceptron_msgs.msg import TrackedObject, TrackedObjectArray

from perceptron_carla import sim_sensors as ss
from perceptron_carla import viz_markers as vm

# ---------------------------------------------------------------- frames
FRAME_WORLD = "carla_world"
FRAME_MAP = "map"
FRAME_ODOM = "odom"
FRAME_FOOTPRINT = "base_footprint"
FRAME_BASE = "base_link"
FRAME_LIDAR = "lidar3d_link"
FRAME_CAMERA = "camera_link"
FRAME_OPTICAL = "camera_optical_link"
FRAME_IMU = "imu_link"
FRAME_GPS = "gps_link"

# ---------------------------------------------------------------- geometry (ROS axes)
# perceptron_robot.xacro: base_footprint -> base_link is +wheel_radius in z.
BASE_LINK_Z = 0.0625
# Sensor poses relative to base_link: (x, y, z, roll, pitch, yaw). x of the camera and
# lidar is replaced at runtime by the rig position (see Mount), everything else is the URDF.
ROVER_SENSORS = {
    FRAME_CAMERA: (0.0, 0.0, 0.42 - BASE_LINK_Z, 0.0, -0.07027, 0.0),   # camera_joint, ArUco-solved pitch
    FRAME_LIDAR: (0.0, 0.0, 0.2775 + 0.075, 0.0, 0.26, 0.0),           # lidar3d_joint, 15 deg down
    FRAME_IMU: (-0.0252393, 0.0066083, -0.0089, 0.0, 0.0, 0.0),         # imu_joint
    FRAME_GPS: (0.0, 0.0, 0.40, 0.0, 0.0, 0.0),                         # not in the URDF: assumed
}
# The original passenger-car layout, kept for comparison runs (--mount car).
CAR_SENSORS = {
    FRAME_CAMERA: (None, 0.0, 1.6 - BASE_LINK_Z, 0.0, 0.0, 0.0),
    FRAME_LIDAR: (None, 0.0, 1.9 - BASE_LINK_Z, 0.0, 0.0, 0.0),
    FRAME_IMU: (0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
    FRAME_GPS: (0.0, 0.0, 1.5, 0.0, 0.0, 0.0),
}
# Rover rig sits this far ahead of the host vehicle's front face, so the low camera
# sees the road and not the bonnet.
RIG_CLEARANCE = 0.10

# CARLA 0.9.15 semantic tags; index = value in /carla/camera/semantic.
SEMANTIC_LABELS = [
    "unlabeled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
    "traffic_sign", "vegetation", "terrain", "sky", "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "static", "dynamic", "other", "water", "road_line", "ground",
    "bridge", "rail_track", "guard_rail"]
# Tags an actor's pixels can carry (props are static/dynamic). Everything else is scenery.
OBJECT_TAGS = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21], dtype=np.uint8)

BOX_COLOURS = {"pedestrian": (60, 60, 255), "traffic_cone": (0, 200, 255), "box": (0, 200, 255)}
VEHICLE_COLOUR = (80, 220, 80)
PROP_COLOUR = (0, 200, 255)

# Covariances copied from the real drivers, so the EKF weighs sim data as it weighs real data.
ODOM_POSE_COV_DIAG = (1e6, 1e6, 1e6, 1e6, 1e6, 1e6)       # stm32_bridge_node POSE_COV_DIAG
ODOM_TWIST_COV_DIAG = (0.01, 1e6, 1e6, 1e6, 1e6, 0.05)    # stm32_bridge_node TWIST_COV_DIAG
IMU_ORIENT_COV = (0.005, 0.005, 0.01)                     # pixhack_bridge_node
IMU_GYRO_COV = 0.001
IMU_ACCEL_COV = 0.05


def carla_tf_to_ros(t: carla.Transform):
    """Returns (x, y, z), (qx, qy, qz, qw), yaw_ros."""
    roll, pitch, yaw = ss.rotation_carla_to_ros_rpy(t.rotation.roll, t.rotation.pitch, t.rotation.yaw)
    return (t.location.x, -t.location.y, t.location.z), ss.quat_from_rpy(roll, pitch, yaw), yaw


def stamp_from_sec(sec: float) -> Time:
    s = int(sec)
    return Time(sec=s, nanosec=int((sec - s) * 1e9))


def diag6(d):
    cov = [0.0] * 36
    for i, v in enumerate(d):
        cov[i * 7] = float(v)
    return cov


def actor_class(a):
    """Detector-style class name for a CARLA actor."""
    tid = a.type_id
    if tid.startswith("walker."):
        return "pedestrian"
    if tid.startswith("vehicle."):
        return a.attributes.get("base_type", "") or "vehicle"
    name = re.sub(r"\d+$", "", tid.split(".")[-1])     # static.prop.trafficcone01 -> trafficcone
    return {"trafficcone": "traffic_cone"}.get(name, name)


def box_colour(cls):
    if cls in BOX_COLOURS:
        return BOX_COLOURS[cls]
    return VEHICLE_COLOUR if cls in ("car", "truck", "van", "bus", "motorcycle", "bicycle",
                                     "vehicle") else PROP_COLOUR


def instance_colours(ids):
    """uint16 ids -> HxWx3 uint8, a stable pseudo-random colour per id, 0 = black."""
    h = ids.astype(np.uint32) * np.uint32(2654435761)
    out = np.stack([(h >> 8) & 255, (h >> 16) & 255, (h >> 24) & 255], axis=-1).astype(np.uint8)
    out[ids == 0] = 0
    return out


def make_image(stamp, arr, encoding):
    """numpy HxW[xC] -> sensor_msgs/Image. array.array('B') skips rclpy's per-byte validation."""
    arr = np.ascontiguousarray(arr)
    m = Image()
    m.header.stamp, m.header.frame_id = stamp, FRAME_OPTICAL
    m.height, m.width, m.encoding, m.is_bigendian = arr.shape[0], arr.shape[1], encoding, 0
    m.step = arr.strides[0]
    m.data = array.array("B", arr.tobytes())
    return m


class Mount:
    """Where every frame sits on the host vehicle.

    base_footprint is the centre of the REAR AXLE on the ground. That is the one point of a
    car-like vehicle with no sideways velocity, which is what the wheel-odometry model and
    the EKF (it fuses vy = 0 from /odom) both assume; the rover's own base_footprint, the
    centre of its skid-steer rectangle, has the same property.
    """

    def __init__(self, vehicle, mount):
        tf = vehicle.get_transform()
        inv = np.array(tf.get_inverse_matrix())
        wheels = vehicle.get_physics_control().wheels
        local = [inv @ np.array([w.position.x / 100.0, w.position.y / 100.0, w.position.z / 100.0, 1.0])
                 for w in wheels]
        self.rear_x = float(np.mean([p[0] for p in local[2:]]))        # CARLA actor frame
        self.front_x = float(vehicle.bounding_box.location.x + vehicle.bounding_box.extent.x)
        self.wheelbase = float(np.mean([p[0] for p in local[:2]]) - self.rear_x)
        self.max_steer_deg = float(wheels[0].max_steer_angle)
        # base_footprint (ROS axes): x from the rear axle, y left
        self.wheels = [(float(p[0]) - self.rear_x, -float(p[1]), w.radius / 100.0)
                       for p, w in zip(local, wheels)]
        self.kind = mount
        if mount == "rover":
            self.rig_x = self.front_x - self.rear_x + RIG_CLEARANCE      # base_footprint frame
            table = ROVER_SENSORS
        else:
            self.rig_x = None
            table = CAR_SENSORS
        self.sensors = {}
        for frame, (x, y, z, r, p, yw) in table.items():
            if frame in (FRAME_CAMERA, FRAME_LIDAR):
                # car layout: camera 1.5 m and lidar 0.0 m ahead of the actor origin
                x = self.rig_x if mount == "rover" else (1.5 if frame == FRAME_CAMERA else 0.0) - self.rear_x
            self.sensors[frame] = (x, y, z, r, p, yw)

    def carla_transform(self, frame):
        """base_link-relative ROS pose -> carla.Transform relative to the actor origin."""
        x, y, z, r, p, yw = self.sensors[frame]
        return carla.Transform(carla.Location(x=x + self.rear_x, y=-y, z=z + BASE_LINK_Z),
                               carla.Rotation(roll=math.degrees(r), pitch=-math.degrees(p),
                                              yaw=-math.degrees(yw)))

    def footprint_world(self, vehicle_tf):
        """carla.Transform of base_footprint in the world."""
        loc = vehicle_tf.transform(carla.Location(x=self.rear_x, y=0.0, z=0.0))
        return carla.Transform(loc, vehicle_tf.rotation)


class CarlaBridge(Node):
    def __init__(self, args):
        super().__init__("carla_bridge")
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)

        def img_pub(topic):
            return self.create_publisher(Image, topic, sensor_qos)

        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_img = img_pub("/carla/camera/image_raw")
        self.pub_info = self.create_publisher(CameraInfo, "/carla/camera/camera_info", 10)
        self.pub_depth = img_pub("/carla/camera/depth")
        self.pub_sem = img_pub("/carla/camera/semantic")
        self.pub_sem_col = img_pub("/carla/camera/semantic_color")
        self.pub_inst = img_pub("/carla/camera/instance")
        self.pub_inst_col = img_pub("/carla/camera/instance_color")
        self.pub_boxes_dbg = img_pub("/carla/camera/boxes_debug")
        self.pub_pc = self.create_publisher(PointCloud2, "/carla/lidar/points", sensor_qos)
        self.pub_pc_gt = self.create_publisher(PointCloud2, "/carla/ground_truth/lidar_semantic", sensor_qos)
        # Robot-equivalent topics: same names and QoS as the real drivers.
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_imu = self.create_publisher(Imu, "/imu/data", sensor_qos)
        self.pub_gps = self.create_publisher(NavSatFix, "/gps/fix", 10)
        self.pub_gt_ego = self.create_publisher(Odometry, "/carla/ground_truth/ego_odom", 10)
        self.pub_gt = self.create_publisher(MarkerArray, "/carla/ground_truth", 10)
        self.pub_b3 = self.create_publisher(Detection3DArray, "/carla/ground_truth/boxes_3d", 10)
        self.pub_gt_obj = self.create_publisher(TrackedObjectArray, "/carla/ground_truth/objects", 10)
        self.pub_b2 = self.create_publisher(Detection2DArray, "/carla/ground_truth/boxes_2d", 10)
        self.pub_b2a = self.create_publisher(Detection2DArray, "/carla/ground_truth/boxes_2d_amodal", 10)
        self.pub_labels = self.create_publisher(String, "/carla/camera/semantic_labels", latched)
        self.pub_labels.publish(String(data=json.dumps(SEMANTIC_LABELS)))
        self.tf_dyn = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)

        # ---------------- connect
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(20.0)
        self.get_logger().info(f"CARLA server {self.client.get_server_version()} "
                               f"/ client {self.client.get_client_version()}")
        if args.town:
            self.world = self.client.load_world(args.town)
        else:
            self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Map-baked parked cars are scenery, not actors: no id, no ground truth. Unload them
        # so every car a detector can see has a box (works on the *_Opt maps).
        self.parked_unloaded = False
        if not args.parked:
            try:
                self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
                self.parked_unloaded = True
            except Exception as exc:
                self.get_logger().warn(f"could not unload parked vehicles: {exc}")

        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.hz
        self.world.apply_settings(settings)

        self.tm = None
        if args.autopilot or args.traffic > 0:
            self.tm = self.client.get_trafficmanager(args.tm_port)
            self.tm.set_synchronous_mode(True)
            self.tm.set_random_device_seed(args.seed)
            self.tm.set_global_distance_to_leading_vehicle(2.5)

        self.actors = []
        self.walker_controllers = []
        bp_lib = self.world.get_blueprint_library()

        # ---------------- ego (host vehicle for the rover sensor rig)
        vehicle_id = args.vehicle or ("vehicle.micro.microlino" if args.mount == "rover"
                                      else "vehicle.lincoln.mkz_2020")
        ego_bp = bp_lib.find(vehicle_id)
        ego_bp.set_attribute("role_name", "ego")
        spawns = self.map.get_spawn_points()
        random.seed(args.seed)
        random.shuffle(spawns)
        self.ego = None
        for sp in spawns:
            self.ego = self.world.try_spawn_actor(ego_bp, sp)
            if self.ego:
                break
        self.actors.append(self.ego)
        for _ in range(10):          # let the vehicle settle on its suspension before measuring
            self.world.tick()
        self.mount = Mount(self.ego, args.mount)
        self.get_logger().info(
            f"ego {vehicle_id}, mount={args.mount}: wheelbase {self.mount.wheelbase:.3f} m, "
            f"max steer {self.mount.max_steer_deg:.0f} deg, rig {self.mount.rig_x} m ahead of "
            f"base_footprint (rear axle)")

        # ---------------- cameras: one pose, one set of intrinsics, so pixel (u, v) means
        # the same ray in every image (RGB, depth, semantic, instance)
        self.img_w, self.img_h, self.fov = args.width, args.height, args.fov
        self.queues = {}
        self.sensors = []
        cam_tf = self.mount.carla_transform(FRAME_CAMERA)

        def spawn(key, bp, transform):
            s = self.world.spawn_actor(bp, transform, attach_to=self.ego)
            self.actors.append(s)
            self.sensors.append(s)
            self.queues[key] = queue.Queue()
            s.listen(self.queues[key].put)

        def camera(key, bp_name):
            bp = bp_lib.find(bp_name)
            for k, v in (("image_size_x", self.img_w), ("image_size_y", self.img_h),
                         ("fov", self.fov), ("sensor_tick", 0.0)):
                bp.set_attribute(k, str(v))
            spawn(key, bp, cam_tf)

        camera("rgb", "sensor.camera.rgb")
        # depth is also what ties instance pixels to actors, so --instance/--boxes need it
        if args.depth or args.instance or args.boxes:
            camera("depth", "sensor.camera.depth")
        if args.semantic:
            camera("semantic", "sensor.camera.semantic_segmentation")
        if args.instance or args.boxes:
            camera("instance", "sensor.camera.instance_segmentation")

        # ---------------- 3D LiDAR (one full sweep per tick)
        rover = args.mount == "rover"
        channels = args.lidar_channels or (16 if rover else 64)
        lidar_geometry = (("channels", channels), ("range", 100), ("rotation_frequency", args.hz),
                          ("points_per_second", args.lidar_pps or (300_000 if rover else 1_300_000)),
                          ("upper_fov", 15), ("lower_fov", -15 if rover else -25), ("sensor_tick", 0.0))
        lid_bp = bp_lib.find("sensor.lidar.ray_cast")
        for k, v in lidar_geometry:
            lid_bp.set_attribute(k, str(v))
        # CARLA's default dropoff_general_rate 0.45 silently discards 45 % of all returns at
        # random; a real spinning lidar returns nearly every ray off a solid surface. Range
        # noise instead (a VLP-16 is specified at +/- 3 cm).
        lid_bp.set_attribute("dropoff_general_rate", str(args.lidar_dropoff))
        lid_bp.set_attribute("noise_stddev", str(args.lidar_noise))
        lid_bp.set_attribute("noise_seed", str(args.seed))
        spawn("lidar", lid_bp, self.mount.carla_transform(FRAME_LIDAR))
        # Ground-truth twin: same pose and ray pattern, but every point carries the semantic
        # tag and the actor id it hit. Evaluation only (cloud_eval_node), never a model input.
        if args.lidar_gt:
            sem_bp = bp_lib.find("sensor.lidar.ray_cast_semantic")
            for k, v in lidar_geometry:
                sem_bp.set_attribute(k, str(v))
            spawn("lidar_gt", sem_bp, self.mount.carla_transform(FRAME_LIDAR))

        # ---------------- IMU / GNSS (CARLA sensors supply truth + gyro/accel noise; the
        # attitude and fix noise models live in sim_sensors.py)
        self.imu_model = self.odom_model = self.gnss_model = None
        if args.imu:
            imu_bp = bp_lib.find("sensor.other.imu")
            for axis in "xyz":
                imu_bp.set_attribute(f"noise_gyro_stddev_{axis}", str(args.gyro_noise))
                imu_bp.set_attribute(f"noise_gyro_bias_{axis}", str(args.gyro_bias))
                imu_bp.set_attribute(f"noise_accel_stddev_{axis}", str(args.accel_noise))
            imu_bp.set_attribute("noise_seed", str(args.seed))
            imu_bp.set_attribute("sensor_tick", "0.0")
            spawn("imu", imu_bp, self.mount.carla_transform(FRAME_IMU))
            self.imu_model = ss.ImuOrientation(
                self.rng, 1.0 / args.hz, yaw_bias=math.radians(args.yaw_bias_deg),
                yaw_walk_std=math.radians(args.yaw_walk_deg), yaw_noise_std=math.radians(args.yaw_noise_deg))
        if args.wheel_odom:
            self.odom_model = ss.WheelOdometry(self.rng, args.odom_v_scale_error, args.odom_w_scale_error,
                                               args.odom_v_noise, args.odom_w_noise)
        if args.gnss:
            g_bp = bp_lib.find("sensor.other.gnss")
            g_bp.set_attribute("sensor_tick", str(1.0 / args.gnss_hz))
            spawn("gnss", g_bp, self.mount.carla_transform(FRAME_GPS))
            self.gnss_model = ss.Gnss(self.rng, args.gnss_bias, args.gnss_noise)
            self.gnss_origin = tuple(float(v) for v in args.gnss_origin.split(","))

        # The EKFs anchor odom/map where the robot is when they start, and the bridge anchors
        # carla_world -> map at the spawn pose, so in ekf mode the ego stands still on its
        # brakes until localization has had time to start -- as a real robot would.
        self.hold = args.hold_seconds if args.hold_seconds >= 0 else (10.0 if args.tf_mode == "ekf" else 0.0)
        self.released = self.hold == 0.0
        if args.autopilot and self.released:
            self.release_ego()
        elif not self.released:
            self.ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            self.get_logger().info(f"holding the ego still for {self.hold:.0f} s of sim time "
                                   f"so localization starts at the anchor pose")
        self.crossers = []
        self.dropouts = parse_dropouts(args.dropout)
        if self.dropouts:
            self.get_logger().info("sensor dropouts (sim s after start): " + ", ".join(
                f"{k} {a:.0f}-{b:.0f} s" for k, a, b in self.dropouts))
        if args.scenario:
            self.spawn_test_scenario(bp_lib)
        if args.crossing > 0:
            self.spawn_crossing(bp_lib, args.crossing)
        if args.traffic > 0:
            self.spawn_traffic(bp_lib, spawns, args.traffic)
        if args.pedestrians > 0:
            self.world.set_pedestrians_cross_factor(0.2)
            n = self.spawn_walkers(bp_lib, args.pedestrians, speed=None)
            self.get_logger().info(f"pedestrians: {n} walking around the map")

        self.world.tick()
        self.start_time = self.world.get_snapshot().timestamp.elapsed_seconds
        self.start_footprint = self.mount.footprint_world(self.ego.get_transform())
        self.publish_static_tf()
        self.publish_bridge_info(latched, vehicle_id)
        self.publish_markers(latched)
        self.camera_info = self.make_camera_info()
        self.fx = self.camera_info.k[0]
        self.cx, self.cy = self.camera_info.k[2], self.camera_info.k[5]

        os.makedirs(args.log_dir, exist_ok=True)
        path = os.path.join(args.log_dir, time.strftime("gt_%Y%m%d_%H%M%S.csv"))
        self.csv_f = open(path, "w", newline="")
        self.csv = csv.writer(self.csv_f)
        # positions in carla_world (ROS axes, ENU)
        self.csv.writerow(["frame", "t", "id", "class", "x", "y", "z", "yaw", "vx", "vy",
                           "ext_x", "ext_y", "ext_z"])
        self.get_logger().info(f"tf-mode={args.tf_mode}; ground truth CSV -> {path}")

    def release_ego(self):
        self.ego.apply_control(carla.VehicleControl(brake=0.0, hand_brake=False))
        if self.args.autopilot:
            self.ego.set_autopilot(True, self.args.tm_port)
            self.tm.vehicle_percentage_speed_difference(self.ego, self.args.ego_speed_diff)
            if self.args.route:
                # Traffic Manager takes these turns at the next junctions, in order: the
                # repeatable "turns" scenario instead of whatever the random route gives
                route = [r.strip().capitalize() for r in self.args.route.split(",") if r.strip()]
                self.tm.set_route(self.ego, route)
                self.get_logger().info(f"ego route at the next junctions: {route}")
        self.released = True

    # ------------------------------------------------------------ spawning
    def spawn_test_scenario(self, bp_lib):
        """Cones/boxes at 5/10/20/40 m, a parked car, and slow walkers near the ego."""
        tf = self.ego.get_transform()
        fwd, right = tf.get_forward_vector(), tf.get_right_vector()

        def ground(loc):
            try:
                lab = self.world.ground_projection(loc + carla.Location(z=5), 20.0)
                if lab:
                    return lab.location
            except Exception:
                pass
            return loc

        def place(bp_name, dist, lateral, yaw_off=0.0, lift=0.0):
            loc = tf.location + fwd * dist + right * lateral
            loc = ground(loc) + carla.Location(z=lift)   # a vehicle at exactly ground height collides
            t = carla.Transform(loc, carla.Rotation(yaw=tf.rotation.yaw + yaw_off))
            a = self.world.try_spawn_actor(bp_lib.find(bp_name), t)
            if a:
                self.actors.append(a)
            else:
                self.get_logger().warn(f"could not spawn {bp_name} at {dist} m")
            return a

        for d in (5, 10, 20, 40):
            place("static.prop.trafficcone01", d, -1.5)
            place("static.prop.box01", d, 1.5)
        car = None
        for dist, lateral in ((15, 3.5), (15, -3.5), (12, 3.5), (18, 3.5), (12, -3.5), (18, -3.5)):
            car = place("vehicle.tesla.model3", dist, lateral, lift=0.3)      # first free spot
            if car:
                car.set_simulate_physics(False)   # parked, never moves
                break

        n = self.spawn_walkers(bp_lib, self.args.walkers, speed=self.args.walker_speed,
                               centre=tf.location, radius=35.0)
        self.get_logger().info(f"scenario: cones/boxes at 5/10/20/40 m, parked car, {n} walkers")

    def spawn_crossing(self, bp_lib, count):
        """Pedestrians pacing across the road ahead of the ego at known distances and speeds.

        No AI controller: each walks a straight line across, turns at +/- half_width, and
        walks back, so its true position and velocity are known at every tick -- the
        repeatable case for lidar clustering (Stage 4) and tracking/prediction (Stages 6-7).
        """
        tf = self.ego.get_transform()
        fwd, right = tf.get_forward_vector(), tf.get_right_vector()
        rig = self.mount.rig_x + self.mount.rear_x if self.mount.rig_x is not None else 0.0
        walker_bps = [b for b in bp_lib.filter("walker.pedestrian.*")]
        half = self.args.crossing_width / 2.0
        first = 5.0
        if self.args.occluder:
            # A van parked on the left 6 m ahead hides the left half of the crossings at 10 m
            # and beyond: pedestrians step out from behind it (the occlusion case). Crossings
            # then start at 10 m so nobody walks through it.
            first = 10.0
            loc = tf.location + fwd * (rig + 6.0) - right * 2.5
            try:
                lab = self.world.ground_projection(loc + carla.Location(z=3), 10.0)
                if lab:
                    loc = lab.location
            except Exception:
                pass
            for name in ("vehicle.mercedes.sprinter", "vehicle.volkswagen.t2", "vehicle.tesla.model3"):
                bps = bp_lib.filter(name)
                van = self.world.try_spawn_actor(bps[0], carla.Transform(
                    loc + carla.Location(z=0.3), tf.rotation)) if bps else None
                if van:
                    van.set_simulate_physics(False)
                    self.actors.append(van)
                    self.get_logger().info(f"occluder: {name} parked 6 m ahead, 2.5 m left")
                    break
            else:
                self.get_logger().warn("could not spawn the occluder")
        for i in range(count):
            dist = first + 5.0 * i                             # 5 (10 with --occluder), +5 ... m ahead
            side = -1.0 if i % 2 == 0 else 1.0                 # alternate starting side
            loc = tf.location + fwd * (rig + dist) + right * (side * half)
            try:
                lab = self.world.ground_projection(loc + carla.Location(z=3), 10.0)
                if lab:
                    loc = lab.location
            except Exception:
                pass
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            w = self.world.try_spawn_actor(bp, carla.Transform(loc + carla.Location(z=1.0)))
            if not w:
                self.get_logger().warn(f"could not spawn crossing pedestrian at {dist:.0f} m")
                continue
            self.actors.append(w)
            speed = 0.8 + 0.2 * (i % 4)                        # 0.8 .. 1.4 m/s
            self.crossers.append({"actor": w, "dir": -side, "speed": speed, "origin": tf.location,
                                  "right": right, "half": half})
        self.get_logger().info(f"crossing: {len(self.crossers)} pedestrians pacing across the road "
                               f"at 5 m steps ahead of the rig")

    def drive_crossers(self):
        for c in self.crossers:
            a = c["actor"]
            off = (a.get_location() - c["origin"]).dot(c["right"])
            if off > c["half"]:
                c["dir"] = -1.0
            elif off < -c["half"]:
                c["dir"] = 1.0
            r = c["right"]
            a.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(r.x * c["dir"], r.y * c["dir"], 0.0), speed=c["speed"]))

    def spawn_walkers(self, bp_lib, count, speed, centre=None, radius=None):
        """AI pedestrians on the navigation mesh. speed=None -> random 1.0-1.6 m/s each."""
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        ctrl_bp = bp_lib.find("controller.ai.walker")
        new = []
        for _ in range(count * 20):
            if len(new) >= count:
                break
            loc = self.world.get_random_location_from_navigation()
            if loc is None or (centre is not None and loc.distance(centre) > radius):
                continue
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            w = self.world.try_spawn_actor(bp, carla.Transform(loc + carla.Location(z=1.0)))
            if w:
                self.actors.append(w)
                new.append(w)
        self.world.tick()
        ctrls = [self.world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w) for w in new]
        self.world.tick()
        for c in ctrls:
            c.start()
            c.go_to_location(self.world.get_random_location_from_navigation())
            c.set_max_speed(speed if speed is not None else random.uniform(1.0, 1.6))
        self.walker_controllers.extend(ctrls)
        return len(new)

    def spawn_traffic(self, bp_lib, spawns, n):
        wheels = (2, 4) if self.args.bikes else (4,)
        bps = [b for b in bp_lib.filter("vehicle.*")
               if int(b.get_attribute("number_of_wheels")) in wheels]
        spawned = 0
        for sp in spawns[1:]:
            if spawned >= n:
                break
            v = self.world.try_spawn_actor(random.choice(bps), sp)
            if v:
                v.set_autopilot(True, self.args.tm_port)
                self.actors.append(v)
                spawned += 1
        self.get_logger().info(f"traffic: {spawned} vehicles on autopilot")

    # ------------------------------------------------------------ static TF / camera model
    def publish_static_tf(self):
        def st(parent, child, xyz, rpy=(0.0, 0.0, 0.0)):
            m = TransformStamped()
            m.header.frame_id, m.child_frame_id = parent, child
            m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = xyz
            (m.transform.rotation.x, m.transform.rotation.y,
             m.transform.rotation.z, m.transform.rotation.w) = ss.quat_from_rpy(*rpy)
            return m

        tfs = [st(FRAME_FOOTPRINT, FRAME_BASE, (0.0, 0.0, BASE_LINK_Z))]
        for frame, (x, y, z, r, p, yw) in self.mount.sensors.items():
            tfs.append(st(FRAME_BASE, frame, (x, y, z), (r, p, yw)))
        # camera (x fwd, y left, z up) -> optical (z fwd, x right, y down),
        # the same rpy(-pi/2, 0, -pi/2) as the URDF's camera_optical_joint
        tfs.append(st(FRAME_CAMERA, FRAME_OPTICAL, (0.0, 0.0, 0.0), (-math.pi / 2, 0.0, -math.pi / 2)))
        if self.args.tf_mode == "gt":
            tfs.append(st(FRAME_WORLD, FRAME_MAP, (0.0, 0.0, 0.0)))
            tfs.append(st(FRAME_MAP, FRAME_ODOM, (0.0, 0.0, 0.0)))
        else:
            # The EKFs start at the origin, with yaw from the IMU's absolute (ENU) heading,
            # so their map/odom frames are ENU anchored where the robot started.
            xyz, _, _ = carla_tf_to_ros(self.start_footprint)
            tfs.append(st(FRAME_WORLD, FRAME_MAP, xyz))
        self.tf_static.sendTransform(tfs)

    def publish_bridge_info(self, qos, vehicle_id):
        """Latched description of how this bridge is running, for tools that must refuse a
        wrong setup (localization_eval_node needs ekf mode) or need the host geometry."""
        bb = self.ego.bounding_box
        r = self.mount.rear_x
        self.host_box = host_box = [bb.location.x - bb.extent.x - r, bb.location.x + bb.extent.x - r,   # base_footprint
                    -bb.extent.y, bb.extent.y, 0.0, bb.location.z + bb.extent.z]
        info = {
            "tf_mode": self.args.tf_mode, "mount": self.args.mount, "vehicle": vehicle_id,
            "ego_id": self.ego.id, "rig_x": self.mount.rig_x, "host_box_base_footprint": host_box,
            "hz": self.args.hz, "lidar_gt": bool(self.args.lidar_gt),
        }
        self.pub_info_json = self.create_publisher(String, "/carla/bridge_info", qos)
        self.pub_info_json.publish(String(data=json.dumps(info)))
        self.get_logger().info(f"host body box in base_footprint [x0 x1 y0 y1 z0 z1]: "
                               f"{[round(v, 3) for v in host_box]}")

    def publish_markers(self, qos):
        """Latched: the ego car (follows TF) and, unless --no-lane-markers, the map's lanes."""
        rig = None
        if self.mount.rig_x is not None:
            rig = {"x": self.mount.rig_x,
                   "camera_z": self.mount.sensors[FRAME_CAMERA][2] + BASE_LINK_Z,
                   "lidar_z": self.mount.sensors[FRAME_LIDAR][2] + BASE_LINK_Z}
        self.pub_ego_marker = self.create_publisher(MarkerArray, "/carla/ego_marker", qos)
        self.pub_ego_marker.publish(vm.ego_marker_array(self.host_box, self.mount.wheels, rig, FRAME_FOOTPRINT))
        if not self.args.lane_markers:
            return
        t0 = time.time()
        lanes = vm.lane_polylines(self.map, step=self.args.lane_step)
        markers = vm.lane_marker_array(lanes, FRAME_WORLD)
        self.pub_lanes = self.create_publisher(MarkerArray, "/carla/map/lanes", qos)
        self.pub_lanes.publish(markers)
        self.get_logger().info(f"lane map: {len(lanes)} lanes, "
                               f"{sum(len(m.points) // 2 for m in markers.markers)} segments "
                               f"({time.time() - t0:.1f} s) -> /carla/map/lanes")

    def make_camera_info(self):
        f = self.img_w / (2.0 * math.tan(math.radians(self.fov) / 2.0))
        # (W-1)/2, not W/2: OpenCV/numpy put integer coords at pixel CENTRES, and UE's
        # optical axis passes through the image centre, i.e. between pixels 639 and 640.
        # W/2 biases every projection by +0.5 px in u and v (measured by projection_check_node).
        cx, cy = (self.img_w - 1) / 2.0, (self.img_h - 1) / 2.0
        ci = CameraInfo()
        ci.header.frame_id = FRAME_OPTICAL
        ci.width, ci.height = self.img_w, self.img_h
        ci.distortion_model = "plumb_bob"
        ci.d = [0.0] * 5
        ci.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return ci

    # ------------------------------------------------------------ main loop
    def step(self):
        frame = self.world.tick()
        snap = self.world.get_snapshot()
        sim_t = snap.timestamp.elapsed_seconds
        stamp = stamp_from_sec(sim_t)
        self.pub_clock.publish(Clock(clock=stamp))
        if not self.released and sim_t - self.start_time >= self.hold:
            self.release_ego()
            self.get_logger().info("hold over, ego released")

        data = {k: self._get(q, frame) for k, q in self.queues.items() if k != "gnss"}
        if data["rgb"] is None or data["lidar"] is None:
            self.get_logger().warn(f"frame {frame}: sensor data missing")
            return
        h, w = self.img_h, self.img_w
        # --dropout: the sensor goes silent (its data is still read, so the queues stay in
        # step); ground truth keeps flowing, so the evaluators see what the stack missed
        drop = {k for k, a, b in self.dropouts if a <= sim_t - self.start_time < b}
        cam_on = "camera" not in drop

        def bgra(d):
            return np.frombuffer(d.raw_data, dtype=np.uint8).reshape(h, w, 4)

        rgb = np.ascontiguousarray(bgra(data["rgb"])[:, :, :3])
        if cam_on:
            self.pub_img.publish(make_image(stamp, rgb, "bgr8"))
            self.camera_info.header.stamp = stamp
            self.pub_info.publish(self.camera_info)

        # depth: CARLA packs normalised depth into R + G*256 + B*65536 over 1000 m
        depth = None
        if data.get("depth") is not None:
            d = bgra(data["depth"])
            # astype: uint8 * 256.0 promotes to float64, and 32FC1 must be 4 bytes per pixel
            depth = ((d[:, :, 2].astype(np.float32) + d[:, :, 1] * 256.0 + d[:, :, 0] * 65536.0)
                     * (1000.0 / 16777215.0)).astype(np.float32)
            if self.args.depth and cam_on:
                self.pub_depth.publish(make_image(stamp, depth, "32FC1"))

        # semantic: class id in R. Read the ids before convert(), which rewrites the buffer.
        if data.get("semantic") is not None and cam_on:
            sem = data["semantic"]
            ids = bgra(sem)[:, :, 2].copy()
            self.pub_sem.publish(make_image(stamp, ids, "mono8"))
            sem.convert(carla.ColorConverter.CityScapesPalette)
            self.pub_sem_col.publish(make_image(stamp, bgra(sem)[:, :, :3], "bgr8"))

        # instance: semantic tag in R, object id in G + B*256
        inst_ids = inst_tags = None
        if data.get("instance") is not None:
            a = bgra(data["instance"])
            inst_tags = a[:, :, 2]
            inst_ids = a[:, :, 1].astype(np.uint16) | (a[:, :, 0].astype(np.uint16) << 8)
            if self.args.instance:
                self.pub_inst.publish(make_image(stamp, inst_ids, "16UC1"))
                self.pub_inst_col.publish(make_image(stamp, instance_colours(inst_ids), "bgr8"))

        # LiDAR (x, y, z, intensity) float32, flip y for ROS
        pts = np.frombuffer(data["lidar"].raw_data, dtype=np.float32).reshape(-1, 4).copy()
        pts[:, 1] *= -1.0
        msg = PointCloud2()
        msg.header.stamp, msg.header.frame_id = stamp, FRAME_LIDAR
        msg.height, msg.width = 1, pts.shape[0]
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(("x", "y", "z", "intensity"))]
        msg.is_bigendian, msg.point_step = False, 16
        msg.row_step, msg.is_dense = 16 * pts.shape[0], True
        msg.data = array.array("B", pts.tobytes())
        if "lidar" not in drop:
            self.pub_pc.publish(msg)
        if data.get("lidar_gt") is not None:
            self.publish_semantic_lidar(stamp, data["lidar_gt"])

        ego_tf = self.ego.get_transform()
        v_fwd, w_z = self.publish_ego(stamp, ego_tf)
        if self.crossers:
            self.drive_crossers()
        if self.args.ego_speed > 0 and self.released and not self.args.autopilot:
            # straight ahead at a set speed: the ego-motion compensation test (static
            # objects must come out at ~0 m/s in odom while the robot moves)
            # plain throttle/brake P-controller: CARLA 0.9.15's Ackermann controller crashed
            # the simulator on the Microlino
            err = self.args.ego_speed - v_fwd
            self.ego.apply_control(carla.VehicleControl(
                throttle=float(min(max(0.5 * err + 0.25, 0.0), 0.6)) if err > -0.1 else 0.0,
                brake=float(min(-err, 1.0)) if err < -0.3 else 0.0, steer=0.0))
        if self.odom_model:
            self.publish_wheel_odom(stamp, v_fwd, w_z)
        if self.imu_model and data.get("imu") is not None and "imu" not in drop:
            self.publish_imu(stamp, data["imu"])
        if self.gnss_model:
            self.publish_gnss(silent="gnss" in drop)

        actors = self.gt_actors()
        self.publish_ground_truth(frame, sim_t, stamp, actors)
        if self.args.boxes and cam_on:           # the stand-in detector runs on the camera
            self.publish_boxes_2d(stamp, data["rgb"].transform, rgb, depth, inst_ids, inst_tags, actors)

    SEM_LIDAR_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("cos", "<f4"),
                                ("obj_idx", "<u4"), ("obj_tag", "<u4")])

    def publish_semantic_lidar(self, stamp, d):
        """x, y, z, obj_idx (actor id, 0 = scenery), obj_tag (semantic class) in lidar3d_link."""
        pts = np.frombuffer(d.raw_data, dtype=self.SEM_LIDAR_DTYPE).copy()
        pts["y"] *= -1.0
        msg = PointCloud2()
        msg.header.stamp, msg.header.frame_id = stamp, FRAME_LIDAR
        msg.height, msg.width = 1, len(pts)
        msg.fields = [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                      PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                      PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                      PointField(name="obj_idx", offset=16, datatype=PointField.UINT32, count=1),
                      PointField(name="obj_tag", offset=20, datatype=PointField.UINT32, count=1)]
        msg.is_bigendian, msg.point_step = False, 24
        msg.row_step, msg.is_dense = 24 * len(pts), True
        msg.data = array.array("B", pts.tobytes())
        self.pub_pc_gt.publish(msg)

    # ------------------------------------------------------------ ego state and robot sensors
    def publish_ego(self, stamp, ego_tf):
        """Ground-truth base_footprint pose (evaluation topic) and, in gt mode, odom -> base TF.

        Returns the true forward speed and yaw rate at base_footprint for the wheel model.
        """
        fp = self.mount.footprint_world(ego_tf)
        xyz, q, yaw = carla_tf_to_ros(fp)
        v, w = self.ego.get_velocity(), self.ego.get_angular_velocity()
        w_z = -math.radians(w.z)                   # deg/s, clockwise -> rad/s, counter-clockwise
        # velocity of the rear-axle point: v_origin + w x r; r is along the body x axis
        vx_w, vy_w = v.x, -v.y
        r = self.mount.rear_x
        vx_w += -w_z * (r * math.sin(yaw))
        vy_w += w_z * (r * math.cos(yaw))
        v_fwd = math.cos(yaw) * vx_w + math.sin(yaw) * vy_w
        v_lat = -math.sin(yaw) * vx_w + math.cos(yaw) * vy_w

        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, FRAME_WORLD, FRAME_FOOTPRINT
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = xyz
        (od.pose.pose.orientation.x, od.pose.pose.orientation.y,
         od.pose.pose.orientation.z, od.pose.pose.orientation.w) = q
        od.twist.twist.linear.x, od.twist.twist.linear.y = v_fwd, v_lat
        od.twist.twist.angular.z = w_z
        self.pub_gt_ego.publish(od)

        if self.args.tf_mode == "gt":
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, FRAME_ODOM, FRAME_FOOTPRINT
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = q
            self.tf_dyn.sendTransform(t)
        return v_fwd, w_z

    def publish_wheel_odom(self, stamp, v_true, w_true):
        """Like stm32_bridge_node: twist only, pose at identity with a 1e6 covariance."""
        v, w = self.odom_model.measure(v_true, w_true)
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, FRAME_ODOM, FRAME_FOOTPRINT
        od.pose.pose.orientation.w = 1.0
        od.twist.twist.linear.x, od.twist.twist.angular.z = float(v), float(w)
        od.pose.covariance = diag6(ODOM_POSE_COV_DIAG)
        twist_cov = list(ODOM_TWIST_COV_DIAG)
        twist_cov[1] = self.args.odom_vy_cov
        od.twist.covariance = diag6(twist_cov)
        self.pub_odom.publish(od)

    def publish_imu(self, stamp, d):
        """Like pixhack_bridge_node, but REP-103: ENU orientation, yaw 0 = east, CCW positive."""
        roll, pitch, yaw = ss.rotation_carla_to_ros_rpy(
            d.transform.rotation.roll, d.transform.rotation.pitch, d.transform.rotation.yaw)
        roll, pitch, yaw = self.imu_model.measure(roll, pitch, yaw)
        m = Imu()
        m.header.stamp, m.header.frame_id = stamp, FRAME_IMU
        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = ss.quat_from_rpy(roll, pitch, yaw)
        o = IMU_ORIENT_COV
        m.orientation_covariance = [o[0], 0.0, 0.0, 0.0, o[1], 0.0, 0.0, 0.0, o[2]]
        gx, gy, gz = ss.gyro_carla_to_ros(d.gyroscope.x, d.gyroscope.y, d.gyroscope.z)
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = gx, gy, gz
        m.angular_velocity_covariance = [IMU_GYRO_COV, 0.0, 0.0, 0.0, IMU_GYRO_COV, 0.0, 0.0, 0.0, IMU_GYRO_COV]
        ax, ay, az = ss.accel_carla_to_ros(d.accelerometer.x, d.accelerometer.y, d.accelerometer.z)
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = ax, ay, az
        m.linear_acceleration_covariance = [IMU_ACCEL_COV, 0.0, 0.0, 0.0, IMU_ACCEL_COV, 0.0,
                                            0.0, 0.0, IMU_ACCEL_COV]
        self.pub_imu.publish(m)

    def publish_gnss(self, silent=False):
        """GNSS ticks at --gnss-hz, slower than the world, so drain whatever has arrived."""
        q = self.queues["gnss"]
        while True:
            try:
                d = q.get_nowait()
            except queue.Empty:
                return
            if silent:
                continue
            # CARLA's own geo-reference puts Town10 at lat 0, lon 0: the equator AND the UTM
            # zone 30/31 boundary, where navsat_transform's UTM conversion jumps as the car
            # crosses it. Re-anchor at --gnss-origin: carla_world is ENU, so x is east, y north.
            east, north = d.transform.location.x, -d.transform.location.y
            lat0, lon0 = self.gnss_origin
            lat, lon = self.gnss_model.measure(*ss.offset_latlon(lat0, lon0, east, north))
            m = NavSatFix()
            m.header.stamp, m.header.frame_id = stamp_from_sec(d.timestamp), FRAME_GPS
            m.status.status, m.status.service = NavSatStatus.STATUS_FIX, NavSatStatus.SERVICE_GPS
            m.latitude, m.longitude, m.altitude = float(lat), float(lon), float(d.altitude)
            var = self.gnss_model.variance
            m.position_covariance = [var, 0.0, 0.0, 0.0, var, 0.0, 0.0, 0.0, 2.0 * var]
            m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            self.pub_gps.publish(m)

    # ------------------------------------------------------------ ground truth
    def gt_actors(self):
        """(actor, class, transform) for every vehicle, walker and prop except the ego."""
        out = []
        for a in self.world.get_actors():
            tid = a.type_id
            if a.id == self.ego.id or not (tid.startswith("vehicle.") or tid.startswith("walker.")
                                            or tid.startswith("static.prop.")):
                continue
            out.append((a, actor_class(a), a.get_transform()))
        return out

    def publish_ground_truth(self, frame, sim_t, stamp, actors):
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        b3 = Detection3DArray()
        b3.header.stamp, b3.header.frame_id = stamp, FRAME_WORLD
        objs = TrackedObjectArray()
        objs.header = b3.header
        for a, cls, tf in actors:
            xyz, q, yaw = carla_tf_to_ros(tf)
            v = a.get_velocity()
            bb = a.bounding_box
            ext = bb.extent
            self.csv.writerow([frame, f"{sim_t:.4f}", a.id, cls, f"{xyz[0]:.3f}", f"{xyz[1]:.3f}",
                               f"{xyz[2]:.3f}", f"{yaw:.4f}", f"{v.x:.3f}", f"{-v.y:.3f}",
                               f"{ext.x:.3f}", f"{ext.y:.3f}", f"{ext.z:.3f}"])
            centre = tf.transform(bb.location)      # box centre in the world
            cx, cy, cz = centre.x, -centre.y, centre.z
            size = (max(2 * ext.x, 0.1), max(2 * ext.y, 0.1), max(2 * ext.z, 0.1))

            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, FRAME_WORLD
            m.ns, m.id, m.type, m.action = cls, a.id, Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, cz
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
            m.scale.x, m.scale.y, m.scale.z = size
            b, g, r = box_colour(cls)
            m.color.r, m.color.g, m.color.b, m.color.a = r / 255.0, g / 255.0, b / 255.0, 0.5
            ma.markers.append(m)

            d = Detection3D()
            d.header = b3.header
            d.id = str(a.id)
            d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = cx, cy, cz
            (d.bbox.center.orientation.x, d.bbox.center.orientation.y,
             d.bbox.center.orientation.z, d.bbox.center.orientation.w) = q
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = size
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = cls, 1.0
            hyp.pose.pose = d.bbox.center
            d.results.append(hyp)
            b3.detections.append(d)

            # same shape as the tracker's output, with the TRUE velocity (Stage 6 scoring)
            o = TrackedObject()
            o.id, o.class_id, o.class_confidence, o.state = a.id, cls, 1.0, TrackedObject.CONFIRMED
            o.pose.pose = d.bbox.center
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = v.x, -v.y, v.z
            o.moving = math.hypot(v.x, v.y) > 0.1
            o.size.x, o.size.y, o.size.z = size
            o.source_detection_id = str(a.id)
            objs.objects.append(o)
        self.pub_gt.publish(ma)
        self.pub_b3.publish(b3)
        self.pub_gt_obj.publish(objs)

    # ------------------------------------------------------------ 2D boxes
    def publish_boxes_2d(self, stamp, cam_tf, rgb, depth, inst_ids, inst_tags, actors):
        """Amodal boxes from the projected 3D box; visible boxes from the instance mask."""
        w, h = self.img_w, self.img_h
        f, cx, cy = self.fx, self.cx, self.cy
        world_to_cam = np.array(cam_tf.get_inverse_matrix())
        cam_to_world = np.array(cam_tf.get_matrix())

        # ---- amodal: project the 8 corners of each 3D box (UE camera: x fwd, y right, z up)
        amodal = {}
        for a, cls, tf in actors:
            verts = np.array([[p.x, p.y, p.z, 1.0] for p in a.bounding_box.get_world_vertices(tf)])
            ue = (world_to_cam @ verts.T).T
            front = ue[:, 0] > 0.1
            if not front.any():
                continue
            dist = float(np.linalg.norm(ue[:, :3].mean(axis=0)))
            if dist > self.args.box_range:
                continue
            u = f * ue[front, 1] / ue[front, 0] + cx
            v = f * -ue[front, 2] / ue[front, 0] + cy
            x0, x1 = max(u.min(), 0.0), min(u.max(), w - 1.0)
            y0, y1 = max(v.min(), 0.0), min(v.max(), h - 1.0)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            amodal[a.id] = (x0, y0, x1, y1, dist, cls, a, tf)

        # ---- visible: group object-tagged pixels by instance id, back-project a few of each
        # instance's pixels with the depth image, and give the instance to the actor whose
        # 3D box holds most of them (instance ids are not actor ids, see module docstring)
        visible = {}
        if inst_ids is not None and depth is not None and amodal:
            cand = [amodal[k] for k in amodal]
            local = np.stack([np.array(c[7].get_inverse_matrix()) for c in cand])     # A,4,4
            bloc = np.array([[c[6].bounding_box.location.x, c[6].bounding_box.location.y,
                              c[6].bounding_box.location.z] for c in cand])            # A,3
            bext = np.array([[c[6].bounding_box.extent.x, c[6].bounding_box.extent.y,
                              c[6].bounding_box.extent.z] for c in cand])              # A,3
            ys, xs = np.nonzero(np.isin(inst_tags, OBJECT_TAGS))
            if ys.size:
                ids = inst_ids[ys, xs]
                uniq, inv, counts = np.unique(ids, return_inverse=True, return_counts=True)
                order = np.argsort(inv, kind="stable")
                starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
                for k in range(len(uniq)):
                    if counts[k] < self.args.min_box_pixels:
                        continue
                    idx = order[starts[k]:starts[k] + counts[k]]
                    s = idx[np.linspace(0, len(idx) - 1, min(16, len(idx))).astype(np.int64)]
                    z = depth[ys[s], xs[s]].astype(np.float64)
                    pts = np.stack([z, (xs[s] - cx) * z / f, -(ys[s] - cy) * z / f, np.ones_like(z)])
                    world = cam_to_world @ pts                                         # 4,S
                    loc = np.einsum("aij,js->ais", local, world)[:, :3, :]             # A,3,S
                    margin = 0.15 + 0.01 * z[None, None, :]
                    inside = (np.abs(loc - bloc[:, :, None]) <= bext[:, :, None] + margin).all(axis=1)
                    hits = inside.sum(axis=1)
                    best = int(np.argmax(hits))
                    if hits[best] * 2 < len(s):
                        continue
                    aid = cand[best][6].id
                    bx = (xs[idx].min(), ys[idx].min(), xs[idx].max(), ys[idx].max())
                    if aid in visible:
                        o = visible[aid]
                        bx = (min(o[0], bx[0]), min(o[1], bx[1]), max(o[2], bx[2]), max(o[3], bx[3]))
                        visible[aid] = bx + (o[4] + int(counts[k]),)
                    else:
                        visible[aid] = bx + (int(counts[k]),)

        def det(aid, cls, x0, y0, x1, y1, score):
            d = Detection2D()
            d.header.stamp, d.header.frame_id = stamp, FRAME_OPTICAL
            d.id = str(aid)
            d.bbox.center.position.x, d.bbox.center.position.y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            d.bbox.size_x, d.bbox.size_y = float(x1 - x0), float(y1 - y0)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = cls, float(score)
            d.results.append(hyp)
            return d

        arr_a, arr_v = Detection2DArray(), Detection2DArray()
        for arr in (arr_a, arr_v):
            arr.header.stamp, arr.header.frame_id = stamp, FRAME_OPTICAL
        dbg = rgb.copy() if self.pub_boxes_dbg.get_subscription_count() > 0 else None
        for aid, (x0, y0, x1, y1, dist, cls, _, _) in sorted(amodal.items(), key=lambda kv: -kv[1][4]):
            arr_a.detections.append(det(aid, cls, x0, y0, x1, y1, 1.0))
            if dbg is not None:
                cv2.rectangle(dbg, (int(x0), int(y0)), (int(x1), int(y1)), (160, 160, 160), 1)
            if aid not in visible:
                continue
            vx0, vy0, vx1, vy1, npx = visible[aid]
            frac = min(1.0, ((vx1 - vx0 + 1) * (vy1 - vy0 + 1)) / max((x1 - x0) * (y1 - y0), 1.0))
            arr_v.detections.append(det(aid, cls, vx0, vy0, vx1, vy1, frac))
            if dbg is not None:
                col = box_colour(cls)
                label = f"{cls} #{aid} {dist:.1f}m"
                cv2.rectangle(dbg, (int(vx0), int(vy0)), (int(vx1), int(vy1)), col, 2)
                cv2.putText(dbg, label, (int(vx0), max(int(vy0) - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(dbg, label, (int(vx0), max(int(vy0) - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        self.pub_b2a.publish(arr_a)
        self.pub_b2.publish(arr_v)
        if dbg is not None:
            self.pub_boxes_dbg.publish(make_image(stamp, dbg, "bgr8"))

    @staticmethod
    def _get(q, frame, timeout=2.0):
        end = time.time() + timeout
        while time.time() < end:
            try:
                d = q.get(timeout=max(0.0, end - time.time()))
            except queue.Empty:
                return None
            if d.frame == frame:
                return d
        return None

    def destroy(self):
        try:
            self.csv_f.close()
        except Exception:
            pass
        for s in self.sensors:
            try:
                s.stop()
            except Exception:
                pass
        for c in self.walker_controllers:
            try:
                c.stop()
            except Exception:
                pass
        self.client.apply_batch([carla.command.DestroyActor(a)
                                 for a in self.walker_controllers + self.actors if a])
        if self.tm:
            self.tm.set_synchronous_mode(False)
        self.world.apply_settings(self.original_settings)
        if self.parked_unloaded:
            try:
                self.world.load_map_layer(carla.MapLayer.ParkedVehicles)
            except Exception:
                pass


DROPOUT_SENSORS = ("lidar", "camera", "imu", "gnss")


def parse_dropouts(spec):
    """"lidar:20:2,camera:35:1.5" -> [("lidar", 20.0, 22.0), ("camera", 35.0, 36.5)]: that
    sensor is silent from 20 s to 22 s of sim time after start-up."""
    out = []
    for item in filter(None, (spec or "").split(",")):
        name, start, dur = item.split(":")
        if name not in DROPOUT_SENSORS:
            raise ValueError(f"--dropout: unknown sensor {name!r}, expected one of {DROPOUT_SENSORS}")
        out.append((name, float(start), float(start) + float(dur)))
    return out


def windows_host():
    """Windows host IP as seen from WSL2 (NAT mode) = default gateway."""
    out = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True).stdout
    return out.split()[2] if out else "localhost"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    flag = argparse.BooleanOptionalAction
    p.add_argument("--host", default=os.environ.get("CARLA_HOST") or windows_host())
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--town", default="", help="e.g. Town10HD_Opt, Town03_Opt (empty = keep current)")
    p.add_argument("--hz", type=float, default=20.0, help="sim rate = camera rate = lidar sweep rate")
    g = p.add_argument_group("vehicle and geometry")
    g.add_argument("--mount", choices=("rover", "car"), default="rover",
                   help="rover: URDF sensor heights on a rig ahead of the bumper; car: 1.6/1.9 m roof layout")
    g.add_argument("--vehicle", default="", help="host blueprint (default microlino for rover, mkz for car)")
    g.add_argument("--tf-mode", choices=("gt", "ekf"), default="gt",
                   help="gt: bridge publishes map->odom->base_footprint from ground truth; "
                        "ekf: leave them to robot_localization (sim_localization.launch.py)")
    g.add_argument("--autopilot", action="store_true", help="ego drives itself (Traffic Manager)")
    g.add_argument("--ego-speed", type=float, default=0.0,
                   help="drive straight ahead at this speed, m/s (no autopilot): ego-motion test")
    g.add_argument("--hold-seconds", type=float, default=-1.0,
                   help="sim seconds the ego stays braked at start; -1 = 10 in ekf mode, 0 in gt mode")
    g.add_argument("--ego-speed-diff", type=float, default=30.0,
                   help="ego drives this %% below the speed limit (rover-like speeds)")
    s = p.add_argument_group("cameras and lidar")
    s.add_argument("--width", type=int, default=1280)
    s.add_argument("--height", type=int, default=720)
    s.add_argument("--fov", type=float, default=90.0)
    s.add_argument("--lidar-channels", type=int, default=0, help="0 = 16 (rover) / 64 (car)")
    s.add_argument("--lidar-pps", type=int, default=0, help="0 = 300k (rover) / 1.3M (car)")
    s.add_argument("--lidar-dropoff", type=float, default=0.0,
                   help="fraction of returns dropped at random (CARLA's own default is 0.45)")
    s.add_argument("--lidar-noise", type=float, default=0.015, help="range noise std, m")
    s.add_argument("--lidar-gt", action=flag, default=True,
                   help="semantic lidar twin -> /carla/ground_truth/lidar_semantic")
    s.add_argument("--depth", action=flag, default=True, help="publish /carla/camera/depth")
    s.add_argument("--semantic", action=flag, default=True, help="semantic segmentation camera")
    s.add_argument("--instance", action=flag, default=True, help="instance segmentation camera")
    s.add_argument("--boxes", action=flag, default=True, help="2D visible/amodal boxes + debug image")
    s.add_argument("--box-range", type=float, default=80.0, help="max distance for 2D boxes, m")
    s.add_argument("--min-box-pixels", type=int, default=30, help="smallest visible instance kept")
    n = p.add_argument_group("robot sensors and noise (sim_sensors.py)")
    n.add_argument("--imu", action=flag, default=True, help="publish /imu/data")
    n.add_argument("--wheel-odom", action=flag, default=True, help="publish /odom")
    n.add_argument("--gnss", action=flag, default=True, help="publish /gps/fix")
    n.add_argument("--gnss-hz", type=float, default=10.0)
    n.add_argument("--gnss-origin", default="19.1334,72.9133",
                   help="lat,lon of carla_world's origin (default IIT Bombay; set your test field)")
    n.add_argument("--gnss-noise", type=float, default=1.0, help="white noise std, m")
    n.add_argument("--gnss-bias", type=float, default=0.5, help="std of the per-run constant offset, m")
    n.add_argument("--gyro-noise", type=float, default=0.005, help="rad/s")
    n.add_argument("--gyro-bias", type=float, default=0.002, help="rad/s")
    n.add_argument("--accel-noise", type=float, default=0.05, help="m/s^2")
    n.add_argument("--yaw-bias-deg", type=float, default=2.0, help="constant heading offset")
    n.add_argument("--yaw-walk-deg", type=float, default=0.05, help="heading random walk, deg/sqrt(s)")
    n.add_argument("--yaw-noise-deg", type=float, default=0.5)
    n.add_argument("--odom-v-scale-error", type=float, default=0.02, help="fraction")
    n.add_argument("--odom-w-scale-error", type=float, default=0.05, help="fraction")
    n.add_argument("--odom-v-noise", type=float, default=0.02, help="m/s")
    n.add_argument("--odom-w-noise", type=float, default=0.01, help="rad/s")
    n.add_argument("--odom-vy-cov", type=float, default=ODOM_TWIST_COV_DIAG[1],
                   help="vy variance on /odom; 1e6 copies stm32_bridge_node (vy unmeasured), "
                        "a small value tells the EKF the wheels do not slip sideways")
    w = p.add_argument_group("world")
    w.add_argument("--parked", action=flag, default=False,
                   help="keep the map's baked-in parked cars (they get no ground truth)")
    w.add_argument("--scenario", action="store_true", help="spawn cones/boxes/parked car/walkers")
    w.add_argument("--walkers", type=int, default=6, help="walkers near the ego in --scenario")
    w.add_argument("--walker-speed", type=float, default=1.0, help="m/s (start slow for tracking)")
    w.add_argument("--crossing", type=int, default=0,
                   help="pedestrians pacing across the road at 5, 10, 15 ... m ahead (ego stays put)")
    w.add_argument("--crossing-width", type=float, default=8.0, help="width they pace across, m")
    w.add_argument("--traffic", type=int, default=0, help="background vehicles on autopilot")
    w.add_argument("--bikes", action=flag, default=False, help="include motorbikes/bicycles in traffic")
    w.add_argument("--pedestrians", type=int, default=0, help="pedestrians walking around the map")
    w.add_argument("--seed", type=int, default=7)
    w.add_argument("--occluder", action="store_true",
                   help="with --crossing: a van parked 6 m ahead hides the left side of the crossings")
    w.add_argument("--route", default="",
                   help="with --autopilot: turns at the next junctions, e.g. Left,Right,Straight")
    w.add_argument("--dropout", default="",
                   help="silence sensors: name:start_s:duration_s,... (lidar, camera, imu, gnss)")
    w.add_argument("--lane-markers", action=flag, default=True,
                   help="publish the map's lane lines on /carla/map/lanes (ground truth, RViz)")
    w.add_argument("--lane-step", type=float, default=1.5, help="lane line sampling, m")
    w.add_argument("--log-dir", default=os.path.expanduser("~/carla_logs"))
    return p.parse_args(argv)


def main(argv=None):
    # ros2 run passes --ros-args ...; keep those for rclpy and give the rest to argparse
    import sys
    argv = sys.argv[1:] if argv is None else argv
    cut = argv.index("--ros-args") if "--ros-args" in argv else len(argv)
    args = parse_args(argv[:cut])
    rclpy.init(args=["carla_bridge"] + argv[cut:])
    node = CarlaBridge(args)
    try:
        while rclpy.ok():
            node.step()
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
