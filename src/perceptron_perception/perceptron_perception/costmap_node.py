#!/usr/bin/env python3
"""Rolling 3D-aware costmap with a dynamic risk layer (roadmap Stage 8).

    /perception/points_obstacles + /perception/points_ground (Stage 4, base_footprint)
    /perception/predicted_objects (Stage 7, odom)        /odom (robot speed)       TF
      -> /costmap/voxels        PointCloud2, occupied voxel centres (x, y in odom, z =
                                height above ground)                     -- layer 1
      -> /costmap/obstacles     OccupancyGrid: height-band projection, 100 / 0 / -1
      -> /costmap/inflated      OccupancyGrid: + robot radius and safety margin -- layer 2
      -> /costmap/dynamic_risk  OccupancyGrid: predicted paths, time-to-encounter weighted
                                                                         -- layer 3
      -> /costmap/combined      OccupancyGrid: max of layers 2 and 3; the planner input
      -> /costmap/stats         Float32MultiArray [ms, voxels, lethal cells, risk cells,
                                moving objects, data age ms]

All grids are in odom, rolling with the robot, stamped with the lidar frame they include.
OccupancyGrid values are the Nav2 cost scaled to 0-100 (100 lethal / inscribed); -1 is
unknown. The layers and their reasons are in costmap.py.

Never blocks on TF (see object_tracker_node): clouds are queued and processed once their
odom transform exists.
"""

import array
import time

import message_filters
import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray

from perceptron_msgs.msg import PredictedObject, PredictedObjectArray
from perceptron_navigation import camera_projection as cp
from perceptron_perception import costmap as cm
from perceptron_perception import lidar_projection as lp


class CostmapNode(Node):

    def __init__(self):
        super().__init__('costmap_node')
        d = self.declare_parameter
        d('global_frame', 'odom')
        d('robot_frame', 'base_footprint')
        d('sensor_frame', 'lidar3d_link')
        d('size', 40.0)
        d('resolution', 0.2)
        d('z_resolution', 0.05)
        d('min_obstacle_height', 0.05)
        d('max_obstacle_height', 0.8)
        d('decay_time', 3.0)
        d('unknown_timeout', 10.0)
        d('inscribed_radius', 0.21)            # the rover: 0.415 x 0.45 m footprint
        d('inflation_radius', 0.6)
        d('cost_scaling', 2.5)
        d('nominal_speed', 1.0)                # robot speed assumed when slower/stopped
        d('exclude_margin', 0.3)               # around moving objects, kept out of layer 1
        d('max_tf_wait', 1.0)
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.frame, self.robot_frame, self.sensor_frame = p('global_frame'), p('robot_frame'), p('sensor_frame')
        self.map = cm.RollingCostmap(
            size=float(p('size')), res=float(p('resolution')), z_res=float(p('z_resolution')),
            min_obstacle_height=float(p('min_obstacle_height')),
            max_obstacle_height=float(p('max_obstacle_height')), decay_time=float(p('decay_time')),
            unknown_timeout=float(p('unknown_timeout')), inscribed_radius=float(p('inscribed_radius')),
            inflation_radius=float(p('inflation_radius')), cost_scaling=float(p('cost_scaling')))
        self.nominal_speed = float(p('nominal_speed'))
        self.exclude_margin = float(p('exclude_margin'))
        self.max_tf_wait = float(p('max_tf_wait'))
        self.predictions = None
        self.robot_speed = 0.0
        self.queue = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.pubs = {name: self.create_publisher(OccupancyGrid, '/costmap/' + name, 1)
                     for name in ('obstacles', 'inflated', 'dynamic_risk', 'combined')}
        self.pub_vox = self.create_publisher(PointCloud2, '/costmap/voxels', qos)
        self.pub_stats = self.create_publisher(Float32MultiArray, '/costmap/stats', 10)
        obst = message_filters.Subscriber(self, PointCloud2, '/perception/points_obstacles', qos_profile=qos)
        grnd = message_filters.Subscriber(self, PointCloud2, '/perception/points_ground', qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer([obst, grnd], queue_size=10, slop=0.01)
        self.sync.registerCallback(lambda o, g: self.queue.append((o, g)))
        self.create_subscription(PredictedObjectArray, '/perception/predicted_objects', self._pred_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_timer(0.01, self._drain)
        self.times = []
        self.create_timer(10.0, self._report)
        self.get_logger().info('costmap %.0f m x %.0f m at %.2f m in %s; robot inscribed %.2f m, inflation %.2f m'
                               % (p('size'), p('size'), p('resolution'), self.frame,
                                  p('inscribed_radius'), p('inflation_radius')))

    def _pred_cb(self, msg):
        self.predictions = msg

    def _odom_cb(self, msg):
        self.robot_speed = abs(msg.twist.twist.linear.x)

    def _drain(self):
        while self.queue:
            obst, grnd = self.queue[0]
            when = Time.from_msg(obst.header.stamp)
            if (self.tf_buffer.can_transform(self.frame, obst.header.frame_id, when) and
                    self.tf_buffer.can_transform(self.frame, self.robot_frame, when)):
                self.queue.pop(0)
                self._process(obst, grnd, when)
                continue
            newest = self.queue[-1][0].header.stamp
            if (newest.sec + newest.nanosec * 1e-9) - when.nanoseconds * 1e-9 > self.max_tf_wait:
                self.queue.pop(0)
                self.get_logger().warn('no TF %s <- %s for a cloud; dropped' % (self.frame, obst.header.frame_id),
                                       throttle_duration_sec=5.0)
                continue
            return

    def _process(self, obst, grnd, when):
        t0 = time.perf_counter()
        stamp = obst.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        M = cp.transform_to_matrix(self.tf_buffer.lookup_transform(self.frame, obst.header.frame_id, when))
        robot = cp.transform_to_matrix(self.tf_buffer.lookup_transform(self.frame, self.robot_frame, when))[:2, 3]
        try:
            S = cp.transform_to_matrix(self.tf_buffer.lookup_transform(obst.header.frame_id, self.sensor_frame, when))
            sensor_bf = S[:3, 3]
        except Exception:
            sensor_bf = np.array([0.0, 0.0, 0.4])

        def to_map(cloud):
            f = lp.cloud_fields(cloud, ['x', 'y', 'z'])
            xyz = np.column_stack([f['x'], f['y'], f['z']]).astype(np.float64)
            xy = cp.transform_points(M, np.column_stack([xyz[:, :2], np.zeros(len(xyz))]))[:, :2]
            return np.column_stack([xy, xyz[:, 2]])          # z stays height above ground

        obstacles, ground = to_map(obst), to_map(grnd)
        sensor = np.r_[cp.transform_points(M, np.array([[sensor_bf[0], sensor_bf[1], 0.0]]))[0, :2], sensor_bf[2]]

        moving = self._moving_objects()
        self.map.recenter(robot)
        self.map.integrate(obstacles, ground, sensor, t,
                           exclude=[(ob['path'][0], ob['radius'] + self.exclude_margin) for ob in moving])
        lethal = self.map.lethal()
        inflated = self.map.inflate(lethal)
        risk, dyn_lethal = self.map.dynamic_risk(moving, robot, self.robot_speed, self.nominal_speed)
        dyn_cost = cm.risk_to_cost(risk, dyn_lethal)
        combined = np.maximum(inflated, dyn_cost)
        known = self.map.known() | lethal | (dyn_cost > 0)

        header = obst.header
        header.frame_id = self.frame
        obstacle_cost = np.where(lethal, cm.LETHAL, 0).astype(np.uint8)
        for name, grid in (('obstacles', obstacle_cost), ('inflated', inflated),
                           ('dynamic_risk', dyn_cost), ('combined', combined)):
            self.pubs[name].publish(self._grid(header, grid, known if name != 'dynamic_risk' else None))
        if self.pub_vox.get_subscription_count():
            self.pub_vox.publish(self._voxels(header))
        ms = (time.perf_counter() - t0) * 1000.0
        age = (self.get_clock().now().nanoseconds * 1e-9 - t) * 1000.0
        self.pub_stats.publish(Float32MultiArray(data=[
            float(ms), float(self.map.occupied().sum()), float(lethal.sum()), float((risk > 0.3).sum()),
            float(len(moving)), float(age)]))
        self.times.append(ms)

    def _moving_objects(self):
        if self.predictions is None:
            return []
        out = []
        for o in self.predictions.objects:
            if o.model == PredictedObject.STATIC or not o.path:
                continue
            out.append({'path': np.array([[q.x, q.y] for q in o.path]), 'path_dt': o.path_dt,
                        'horizons': list(o.horizons),
                        'covs': np.array([np.array(q.covariance).reshape(6, 6)[:2, :2] for q in o.poses]),
                        'radius': o.radius})
        return out

    def _grid(self, header, cost, known):
        g = OccupancyGrid()
        g.header = header
        g.info.map_load_time = header.stamp
        g.info.resolution = float(self.map.res)
        g.info.width, g.info.height = self.map.nx, self.map.ny
        g.info.origin.position.x, g.info.origin.position.y = float(self.map.origin[0]), float(self.map.origin[1])
        g.info.origin.orientation.w = 1.0
        g.data = array.array('b', cm.to_occupancy(cost, known).ravel().tobytes())
        return g

    def _voxels(self, header):
        i, j, k = np.nonzero(self.map.occupied())
        xyz = np.column_stack([self.map.origin[0] + (i + 0.5) * self.map.res,
                               self.map.origin[1] + (j + 0.5) * self.map.res,
                               (k + 0.5) * self.map.z_res]).astype(np.float32)
        m = PointCloud2()
        m.header = header
        m.height, m.width = 1, len(xyz)
        m.fields = [PointField(name=n, offset=4 * q, datatype=PointField.FLOAT32, count=1)
                    for q, n in enumerate(('x', 'y', 'z'))]
        m.is_bigendian, m.point_step = False, 12
        m.row_step, m.is_dense = 12 * len(xyz), True
        m.data = array.array('B', xyz.tobytes())
        return m

    def _report(self):
        if not self.times:
            self.get_logger().warn('no obstacle + ground cloud pairs processed', throttle_duration_sec=30.0)
            return
        a = np.array(self.times)
        self.get_logger().info('%d updates, %.1f ms mean, p95 %.1f ms; %d occupied voxels, %d lethal cells'
                               % (len(a), a.mean(), np.percentile(a, 95), int(self.map.occupied().sum()),
                                  int(self.map.lethal().sum())))
        self.times = []


def main(args=None):
    rclpy.init(args=args)
    node = CostmapNode()
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
