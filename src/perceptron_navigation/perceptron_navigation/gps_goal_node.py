#!/usr/bin/env python3
"""Drive to a GPS coordinate with Nav2 (Stage 11).

    /gps_goal           sensor_msgs/NavSatFix   go to (latitude, longitude)
    /gps_goal/relative  geometry_msgs/PointStamped  go to a point given in any frame
                        (e.g. base_footprint: "40 m ahead"); its GPS is logged
    /gps_goal/cancel    std_msgs/Empty           stop and drop the goal
    /clicked_point      geometry_msgs/PointStamped (RViz "Publish Point"): logs the point's
                        GPS, and goes there when click_goes is true
      -> NavigateToPose sub-goals along the route (geo.Route), /gps_goal/route (Path),
         /gps_goal/status (String JSON: state, distance left, sub-goal, attempts)

GPS -> map: the datum (datum_lat, datum_lon) is the origin of `datum_frame`, whose axes are
ENU, and TF gives datum_frame -> map. In CARLA datum_frame is carla_world and the datum the
bridge's --gnss-origin. On the rover, navsat_transform with a fixed datum makes `map` exactly
that, so datum_frame: map.

Route: route_mode 'straight' is the line from the robot to the goal; Nav2 plans around what
the costmap shows. route_mode 'roads' is the shortest path over a road graph (lane-centre
polylines, String JSON on `road_graph_topic`: in CARLA the bridge's /carla/map/road_graph),
any lane, either direction (no traffic rules); it falls back to 'straight' without a graph.
Sub-goals every `subgoal_spacing` m keep each Nav2 goal inside its rolling costmaps.
"""

import json
import math

import numpy as np
import rclpy
import tf2_ros
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Empty, String

from perceptron_navigation import camera_projection as cp
from perceptron_navigation import geo


def yaw_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class GpsGoalNode(Node):

    def __init__(self):
        super().__init__('gps_goal_node')
        d = self.declare_parameter
        d('datum_lat', 19.1334)
        d('datum_lon', 72.9133)
        d('datum_frame', 'carla_world')
        d('map_frame', 'map')
        d('robot_frame', 'base_footprint')
        d('route_mode', 'straight')            # straight | roads
        d('road_graph_topic', '/carla/map/road_graph')
        d('subgoal_spacing', 30.0)
        d('advance_radius', 12.0)
        d('goal_tolerance', 2.5)
        d('max_attempts', 3)
        d('click_goes', True)
        p = lambda n: self.get_parameter(n).value    # noqa: E731
        self.datum = (float(p('datum_lat')), float(p('datum_lon')))
        self.datum_frame, self.map_frame, self.robot_frame = p('datum_frame'), p('map_frame'), p('robot_frame')
        self.spacing, self.advance = float(p('subgoal_spacing')), float(p('advance_radius'))
        self.tolerance, self.max_attempts = float(p('goal_tolerance')), int(p('max_attempts'))
        self.click_goes = bool(p('click_goes'))
        self.route_mode = p('route_mode')
        self.graph_msg = None
        self.graph = None
        if self.route_mode == 'roads':
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            self.create_subscription(String, p('road_graph_topic'), self._graph_cb, latched)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_subscription(NavSatFix, '/gps_goal', self._gps_cb, 10)
        self.create_subscription(PointStamped, '/gps_goal/relative', self._relative_cb, 10)
        self.create_subscription(Empty, '/gps_goal/cancel', lambda _: self._cancel('cancelled'), 10)
        self.create_subscription(PointStamped, '/clicked_point', self._clicked_cb, 10)
        self.pub_route = self.create_publisher(Path, '/gps_goal/route', 1)
        self.pub_status = self.create_publisher(String, '/gps_goal/status', 10)

        self.route = None
        self.goal_latlon = None
        self.goal_handle = None
        self.goal_seq = 0
        self.state = 'idle'
        self.attempts = 0
        self.started = None
        self.create_timer(0.5, self._tick)
        self.get_logger().info('GPS goals: datum %.6f, %.6f at the origin of %s; route %s, sub-goals every %.0f m'
                               % (self.datum[0], self.datum[1], self.datum_frame, p('route_mode'), self.spacing))

    # ------------------------------------------------------------ frames
    def _M(self, target, source):
        return cp.transform_to_matrix(self.tf_buffer.lookup_transform(target, source, Time()))

    def _robot_xy(self):
        M = self._M(self.map_frame, self.robot_frame)
        return M[0, 3], M[1, 3]

    def latlon_to_map(self, lat, lon):
        e, n = geo.latlon_to_enu(lat, lon, *self.datum)
        M = self._M(self.map_frame, self.datum_frame)
        q = cp.transform_points(M, np.array([[e, n, 0.0]]))[0]
        return float(q[0]), float(q[1])

    def point_to_latlon(self, frame, x, y, z=0.0):
        M = self._M(self.datum_frame, frame)
        q = cp.transform_points(M, np.array([[x, y, z]], dtype=float))[0]
        return geo.enu_to_latlon(q[0], q[1], *self.datum)

    def _graph_cb(self, msg):
        self.graph_msg = json.loads(msg.data)
        self.graph = None                              # (re)built in map frame on the next goal
        self.get_logger().info('road graph: %d lanes in %s'
                               % (len(self.graph_msg['lanes']), self.graph_msg['frame']))

    def _road_graph(self):
        if self.graph is None and self.graph_msg is not None:
            M = self._M(self.map_frame, self.graph_msg['frame'])
            lanes = []
            for poly in self.graph_msg['lanes']:
                xy = np.asarray(poly, dtype=float).reshape(-1, 2)
                q = cp.transform_points(M, np.column_stack([xy, np.zeros(len(xy))]))
                lanes.append([tuple(r) for r in q[:, :2]])
            self.graph = geo.RoadGraph(lanes)
            self.get_logger().info('road graph ready: %d points' % len(self.graph.p))
        return self.graph

    def _plan_route(self, start, goal):
        if self.route_mode == 'roads':
            g = self._road_graph()
            if g is None:
                self.get_logger().warn('no road graph yet: straight route')
            else:
                pts = g.route(start, goal)
                if pts is not None:
                    return pts
                self.get_logger().warn('goal not connected to the robot by road: straight route')
        return [start, goal]

    # ------------------------------------------------------------ inputs
    def _gps_cb(self, msg):
        self.start(msg.latitude, msg.longitude)

    def _relative_cb(self, msg):
        try:
            lat, lon = self.point_to_latlon(msg.header.frame_id or self.robot_frame,
                                            msg.point.x, msg.point.y, msg.point.z)
        except Exception as exc:
            self.get_logger().error('cannot place the relative goal: %s' % exc)
            return
        self.get_logger().info('relative goal (%.1f, %.1f) in %s = GPS %.7f, %.7f'
                               % (msg.point.x, msg.point.y, msg.header.frame_id, lat, lon))
        self.start(lat, lon)

    def _clicked_cb(self, msg):
        try:
            lat, lon = self.point_to_latlon(msg.header.frame_id, msg.point.x, msg.point.y, msg.point.z)
        except Exception as exc:
            self.get_logger().error('cannot convert the clicked point: %s' % exc)
            return
        self.get_logger().info('clicked point = GPS %.7f, %.7f   (ros2 run perceptron_navigation gps_goal %.7f %.7f)'
                               % (lat, lon, lat, lon))
        if self.click_goes:
            self.start(lat, lon)

    # ------------------------------------------------------------ goal handling
    def start(self, lat, lon):
        try:
            gx, gy = self.latlon_to_map(lat, lon)
            rx, ry = self._robot_xy()
        except Exception as exc:
            self.get_logger().error('no TF yet (%s); goal ignored' % exc)
            return
        self.goal_latlon = (lat, lon)
        self.route = geo.Route(self._plan_route((rx, ry), (gx, gy)), self.spacing, self.advance)
        self.attempts = 0
        self.started = self.get_clock().now()
        self.state = 'driving'
        self._publish_route()
        self.get_logger().info('GPS goal %.7f, %.7f -> map (%.1f, %.1f), %.0f m in a line, route %.0f m'
                               % (lat, lon, gx, gy, math.hypot(gx - rx, gy - ry), self.route.length))
        self._send(self.route.next_subgoal((rx, ry)))

    def _send(self, pose):
        if pose is None:
            return
        if not self.nav.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('navigate_to_pose is not available (is Nav2 up?)')
            self.state = 'no nav2'
            return
        x, y, yaw = pose
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        (goal.pose.pose.orientation.x, goal.pose.pose.orientation.y,
         goal.pose.pose.orientation.z, goal.pose.pose.orientation.w) = yaw_quat(yaw)
        self.goal_seq += 1
        seq = self.goal_seq
        final = self.route.is_final()
        self.get_logger().info('%s (%.1f, %.1f)' % ('final goal' if final else 'sub-goal', x, y))
        fut = self.nav.send_goal_async(goal)
        fut.add_done_callback(lambda f, seq=seq: self._accepted(f, seq))

    def _accepted(self, fut, seq):
        handle = fut.result()
        if seq != self.goal_seq:
            return
        if not handle.accepted:
            self.get_logger().warn('Nav2 rejected the goal')
            self.state = 'rejected'
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f, seq=seq: self._result(f, seq))

    def _result(self, fut, seq):
        if seq != self.goal_seq or self.route is None:
            return                                     # superseded by a newer sub-goal
        status = fut.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            if self.route.is_final():
                self._finish('reached')
            return                                     # an intermediate one: _tick moves on
        if status == GoalStatus.STATUS_CANCELED:
            return
        self.attempts += 1
        if self.attempts >= self.max_attempts:
            self._finish('failed')
            return
        self.get_logger().warn('Nav2 gave up on the sub-goal (status %d); retry %d of %d'
                               % (status, self.attempts, self.max_attempts - 1))
        self.route.subgoal_s = None
        try:
            self._send(self.route.next_subgoal(self._robot_xy()))
        except Exception:
            pass

    def _cancel(self, why):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self._finish(why)

    def _finish(self, state):
        secs = (self.get_clock().now() - self.started).nanoseconds * 1e-9 if self.started else 0.0
        left = self._distance_left()
        self.get_logger().info('GPS goal %s after %.1f s, %.1f m from it' % (state.upper(), secs, left or -1))
        self.state = state
        self.route = None
        self.goal_seq += 1                             # ignore late results
        self._publish_status()

    def _distance_left(self):
        if self.goal_latlon is None:
            return None
        try:
            gx, gy = self.latlon_to_map(*self.goal_latlon)
            rx, ry = self._robot_xy()
            return math.hypot(gx - rx, gy - ry)
        except Exception:
            return None

    def _tick(self):
        if self.route is not None and self.state == 'driving':
            try:
                xy = self._robot_xy()
            except Exception:
                return
            left = self._distance_left()
            if self.route.is_final() and left is not None and left <= self.tolerance:
                self._cancel('reached')               # close enough: Nav2's yaw check is moot for a car
                return
            self._send(self.route.next_subgoal(xy))
        self._publish_status()

    def _publish_route(self):
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.route.p:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_route.publish(path)

    def _publish_status(self):
        left = self._distance_left()
        st = {'state': self.state, 'distance_left': None if left is None else round(left, 2),
              'goal': self.goal_latlon, 'attempts': self.attempts}
        if self.route is not None and self.route.subgoal_s is not None:
            st['subgoal_s'] = round(self.route.subgoal_s, 1)
            st['route_length'] = round(self.route.length, 1)
        self.pub_status.publish(String(data=json.dumps(st)))


def main(args=None):
    rclpy.init(args=args)
    node = GpsGoalNode()
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
