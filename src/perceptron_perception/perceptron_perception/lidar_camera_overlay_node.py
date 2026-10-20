#!/usr/bin/env python3
"""Draw the 3D lidar cloud onto the camera image (Task 01, geometric projection).

    /points (lidar3d_link) --TF--> camera_optical_link --cull--> projectPoints
    /camera/image_raw  ---------------------------------------> dots, coloured by depth
                                                               -> /perception/lidar_overlay

The geometry is perceptron_navigation.camera_projection, exactly what
path_overlay_node uses for the Nav2 plan: same URDF chain (whose camera pitch
was solved from the ArUco boards), same intrinsics, same two-stage cull. So if
the plan ribbon sits on the floor in the real video, the lidar does too, and
a misalignment seen here is in the lidar's extrinsic, not the camera's.

INTRINSICS: from camera_info_topic when it is set (CARLA, or any driver that
publishes one), otherwise from the camera_matrix / dist_coeffs parameters --
the measured camera_calib.json values -- rescaled to the image size.

TF: the lidar-to-camera transform is static, so the latest one is the right
one; asking for it at the cloud's stamp would only add a failure mode. Image
and cloud are paired by stamp with an approximate-time synchroniser.

ON THE REAL ROBOT the camera does not join the ROS graph (see path_overlay_node:
the Jetson owns /dev/video0 and encodes straight to H.265). This node is for
the simulator and for bags that contain an image topic.
"""

import array

import message_filters
import numpy as np
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from perceptron_navigation import camera_projection as cp
from perceptron_navigation import marker_map as mm
from perceptron_perception import lidar_projection as lp


class LidarCameraOverlayNode(Node):

    def __init__(self):
        super().__init__('lidar_camera_overlay_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('cloud_topic', '/points')
        self.declare_parameter('camera_info_topic', '')
        self.declare_parameter('output_topic', '/perception/lidar_overlay')
        self.declare_parameter('camera_optical_frame', 'camera_optical_link')

        # Same measured values as path_overlay.yaml / aruco_localization.yaml.
        self.declare_parameter('camera_matrix', [
            410.9212916385953, 0.0, 634.8042096493966,
            0.0, 411.4395116912116, 347.58359871233904,
            0.0, 0.0, 1.0])
        self.declare_parameter('dist_coeffs', [
            0.030287865182229395, -0.002860707267771121,
            -0.004223044189411946, -0.0003374289867825264,
            -0.0003820853552867496])
        self.declare_parameter('calibration_width', 1280)
        self.declare_parameter('calibration_height', 720)

        self.declare_parameter('min_z', 0.15)
        self.declare_parameter('max_radius', 2.35)
        self.declare_parameter('max_range', 40.0)
        self.declare_parameter('colour_max_depth', 30.0)
        self.declare_parameter('point_radius', 1)
        self.declare_parameter('sync_slop', 0.05)
        # Optical-frame offset applied on top of TF, for calibration-sensitivity tests.
        self.declare_parameter('perturb_rpy_deg', [0.0, 0.0, 0.0])
        self.declare_parameter('perturb_xyz', [0.0, 0.0, 0.0])

        p = self.get_parameter
        self.optical_frame = p('camera_optical_frame').value
        self.base_K = np.array(p('camera_matrix').value, dtype=np.float64).reshape(3, 3)
        self.base_D = np.array(p('dist_coeffs').value, dtype=np.float64)
        self.calib_size = (int(p('calibration_width').value), int(p('calibration_height').value))
        self.min_z = float(p('min_z').value)
        self.max_radius = float(p('max_radius').value)
        self.max_range = float(p('max_range').value)
        self.colour_max_depth = float(p('colour_max_depth').value)
        self.point_radius = int(p('point_radius').value)
        self.perturb = lp.perturbation(p('perturb_rpy_deg').value, p('perturb_xyz').value)

        self.camera_info = None
        info_topic = p('camera_info_topic').value
        if info_topic:
            self.create_subscription(CameraInfo, info_topic, self._info_cb, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Image, p('output_topic').value, sensor_qos)
        img_sub = message_filters.Subscriber(self, Image, p('image_topic').value,
                                             qos_profile=sensor_qos)
        pc_sub = message_filters.Subscriber(self, PointCloud2, p('cloud_topic').value,
                                            qos_profile=sensor_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, pc_sub], queue_size=10, slop=float(p('sync_slop').value))
        self.sync.registerCallback(self._pair_cb)

        self.frames = 0
        self.points_drawn = 0
        self.reports = 0
        self.create_timer(10.0, self._report)
        self.get_logger().info(
            'Lidar overlay: %s + %s -> %s (intrinsics from %s)'
            % (p('image_topic').value, p('cloud_topic').value, p('output_topic').value,
               info_topic or 'parameters'))

    def _info_cb(self, msg):
        self.camera_info = msg

    def _intrinsics(self, width, height):
        if self.camera_info is not None:
            K, D, _ = lp.intrinsics_from_camera_info(self.camera_info)
            return K, D
        return mm.scale_intrinsics(self.base_K, self.calib_size, (width, height)), self.base_D

    def _pair_cb(self, img_msg, cloud_msg):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.optical_frame, cloud_msg.header.frame_id, rclpy.time.Time())
        except Exception as exc:
            self.get_logger().warn('No transform %s -> %s (%s)'
                                   % (cloud_msg.header.frame_id, self.optical_frame, exc),
                                   throttle_duration_sec=10.0)
            return
        M = self.perturb @ cp.transform_to_matrix(tf)

        img = lp.image_to_bgr(img_msg)
        h, w = img.shape[:2]
        K, D = self._intrinsics(w, h)

        pts_opt = cp.transform_points(M, lp.cloud_to_xyz(cloud_msg))
        pts_opt = pts_opt[cp.visible_mask(pts_opt, self.min_z, self.max_radius, self.max_range)]
        uv = cp.project_points(pts_opt, K, D)
        on = cp.inside_image(uv, w, h)
        uv, z = uv[on], pts_opt[on, 2]
        lp.draw_points(img, uv, lp.depth_colours(z, self.colour_max_depth), z, self.point_radius)

        out = Image()
        out.header = img_msg.header
        out.height, out.width, out.encoding, out.is_bigendian = h, w, 'bgr8', 0
        out.step = w * 3
        # array.array, not bytes: rclpy validates a bytes value one element at a
        # time, which alone caps a 1280x720 stream at a few Hz.
        out.data = array.array('B', img.tobytes())
        self.pub.publish(out)
        self.frames += 1
        self.points_drawn += int(on.sum())

    def _report(self):
        self.reports += 1
        if self.frames == 0 and self.reports == 1:
            # Under use_sim_time the first tick fires on the /clock jump at startup,
            # before any pair could have arrived; it says nothing.
            return
        if self.frames == 0:
            self.get_logger().warn('No image+cloud pairs in the last 10 s -- check topics, '
                                   'QoS and that the stamps are within sync_slop.')
        else:
            self.get_logger().info('%d overlays, %.0f points/frame'
                                   % (self.frames, self.points_drawn / self.frames))
        self.frames = 0
        self.points_drawn = 0


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraOverlayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
