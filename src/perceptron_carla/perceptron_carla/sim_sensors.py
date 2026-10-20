"""Simulated rover sensors: CARLA truth -> noisy measurements shaped like the real ones.

No ROS or CARLA imports, so the models are testable on their own -- see
test/test_sim_sensors.py. carla_bridge.py feeds them truth and publishes what
they return on the robot's real topic names.

WHAT EACH MODEL STANDS IN FOR (and what the real robot publishes)

  wheel odometry  stm32_bridge_node /odom: twist only (vx, wz) with the pose
                  left at identity and a 1e6 covariance, so robot_localization
                  treats it as unmeasured. Errors: a scale factor on each axis
                  (tyre radius, skid-steer scrub) plus white noise. Never the
                  true pose -- that would hand the EKF the answer.
  IMU             pixhack_bridge_node /imu/data: an absolute orientation from
                  the flight controller's own filter, body rates, and raw
                  acceleration including gravity. Errors: the yaw carries a
                  constant offset plus a slow random walk (a compass is never
                  exactly right), roll/pitch carry white noise; the gyro bias
                  and noise come from CARLA's own IMU sensor attributes.
  GNSS            pixhack_bridge_node /gps/fix. Errors: a constant offset
                  plus white noise, in metres, east/north.

CONVENTIONS (measured on CARLA 0.9.15, not assumed)

CARLA is left-handed: x forward, y RIGHT, z up, yaw clockwise. ROS (REP 103)
is x forward, y left, z up, yaw counter-clockwise. Flipping y converts a
position; a vector of rates is a pseudovector and also flips x and z. CARLA's
+x is east and +y is south, so with y flipped the world is ENU, which is what
navsat_transform expects with yaw_offset 0. CARLA's accelerometer includes
gravity (+9.81 on z at rest), like the real Pixhack.
"""

import math

import numpy as np

EARTH_M_PER_DEG = 111320.0


# ------------------------------------------------------------ conversions

def accel_carla_to_ros(x, y, z):
    return x, -y, z


def gyro_carla_to_ros(x, y, z):
    """Body rates: a pseudovector, so mirroring y flips the x and z components."""
    return -x, y, -z


def rotation_carla_to_ros_rpy(roll_deg, pitch_deg, yaw_deg):
    return math.radians(roll_deg), -math.radians(pitch_deg), -math.radians(yaw_deg)


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def offset_latlon(lat, lon, east_m, north_m):
    """Move a WGS84 point by a small east/north offset in metres."""
    dlat = north_m / EARTH_M_PER_DEG
    dlon = east_m / (EARTH_M_PER_DEG * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


# ------------------------------------------------------------ noise models

class WheelOdometry:
    """True forward speed and yaw rate -> what the wheel encoders would report."""

    def __init__(self, rng, v_scale_error=0.02, w_scale_error=0.05,
                 v_noise_std=0.02, w_noise_std=0.01):
        self.rng = rng
        self.v_scale = 1.0 + v_scale_error
        self.w_scale = 1.0 + w_scale_error
        self.v_std = v_noise_std
        self.w_std = w_noise_std

    def measure(self, v_true, w_true):
        v = v_true * self.v_scale + self.rng.normal(0.0, self.v_std)
        w = w_true * self.w_scale + self.rng.normal(0.0, self.w_std)
        return v, w


class ImuOrientation:
    """True roll/pitch/yaw (ROS, radians) -> a flight controller's attitude estimate."""

    def __init__(self, rng, dt, yaw_bias=math.radians(2.0), yaw_walk_std=math.radians(0.05),
                 yaw_noise_std=math.radians(0.5), tilt_noise_std=math.radians(0.3)):
        self.rng = rng
        self.yaw_offset = yaw_bias
        # random-walk step scaled so yaw_walk_std is the drift per sqrt(second)
        self.walk_step = yaw_walk_std * math.sqrt(dt)
        self.yaw_std = yaw_noise_std
        self.tilt_std = tilt_noise_std

    def measure(self, roll, pitch, yaw):
        self.yaw_offset += self.rng.normal(0.0, self.walk_step)
        return (roll + self.rng.normal(0.0, self.tilt_std),
                pitch + self.rng.normal(0.0, self.tilt_std),
                wrap_angle(yaw + self.yaw_offset + self.rng.normal(0.0, self.yaw_std)))


class Gnss:
    """True lat/lon -> a fix with a fixed offset plus white noise, in metres."""

    def __init__(self, rng, bias_std=0.5, noise_std=1.0):
        self.rng = rng
        self.bias = rng.normal(0.0, bias_std, size=2)       # east, north
        self.noise_std = noise_std
        self.variance = noise_std ** 2 + bias_std ** 2

    def measure(self, lat, lon):
        e, n = self.bias + self.rng.normal(0.0, self.noise_std, size=2)
        return offset_latlon(lat, lon, e, n)
