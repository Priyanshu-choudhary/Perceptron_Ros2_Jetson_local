"""ROS velocity commands -> CARLA throttle / brake / steering for the car-like host (no ROS).

Nav2 (and the rover's own stack) speaks geometry_msgs/Twist: forward speed v and yaw rate w.
The Microlino in CARLA has Ackermann steering, so the command becomes

    curvature  k = w / v                       (v floored at v_min_steer: a car cannot turn
    steering   d = atan(wheelbase * k)          on the spot, so a pure rotation means "turn
                                                as tightly as allowed while creeping")
    CARLA steer = -d / max_wheel_steer          (CARLA: +1 = full right; ROS: +w = left)

and the speed is held by a PI loop on the measured forward speed, with a feed-forward
throttle, braking when too fast or told to stop. CARLA 0.9.15's own Ackermann controller is
not used: it crashed the simulator on this car (Stage 1).

Safety, in this order:
  e-stop or a stale command (older than `timeout`)  -> brake to a stop and hold
  limits                                            -> |v| <= max_speed (max_reverse backwards),
                                                       |d| <= max_steer, steer rate limited
  direction change                                  -> brake to ~0 before switching gear
"""

import math
from dataclasses import dataclass


@dataclass
class Control:
    throttle: float = 0.0
    steer: float = 0.0          # CARLA convention, -1 (left) .. +1 (right)
    brake: float = 0.0
    reverse: bool = False
    state: str = 'idle'


class TwistController:

    def __init__(self, wheelbase, max_wheel_steer, max_steer=math.radians(35.0), max_speed=4.0,
                 max_reverse=1.0, timeout=0.5, kp=0.35, ki=0.25, ff0=0.18, ff1=0.035,
                 max_throttle=0.75, steer_rate=1.5, v_min_steer=0.5):
        self.L = float(wheelbase)
        self.max_wheel_steer = float(max_wheel_steer)   # rad of wheel angle at CARLA steer = 1
        self.max_steer = min(float(max_steer), self.max_wheel_steer)
        self.max_speed, self.max_reverse = float(max_speed), float(max_reverse)
        self.timeout = float(timeout)
        self.kp, self.ki, self.ff0, self.ff1 = kp, ki, ff0, ff1
        self.max_throttle = max_throttle
        self.steer_rate = steer_rate                    # CARLA steer units per second
        self.v_min_steer = v_min_steer
        self.integral = 0.0
        self.steer = 0.0
        self.reverse = False

    def min_turning_radius(self):
        return self.L / math.tan(self.max_steer)

    def step(self, v_cmd, w_cmd, v_meas, dt, cmd_age, estop=False):
        """One control tick. v_meas: signed forward speed (m/s); cmd_age: s since the last
        command (inf if none yet)."""
        out = Control(reverse=self.reverse)
        if estop or cmd_age > self.timeout:
            v_cmd, w_cmd = 0.0, 0.0
            out.state = 'estop' if estop else 'stale'
        else:
            out.state = 'tracking'
        v_t = max(-self.max_reverse, min(self.max_speed, float(v_cmd)))

        # steering from the commanded curvature
        v_ref = v_t if abs(v_t) >= self.v_min_steer else math.copysign(self.v_min_steer, v_t if v_t else 1.0)
        d = math.atan(self.L * w_cmd / v_ref) if w_cmd else 0.0
        d = max(-self.max_steer, min(self.max_steer, d))
        target = -d / self.max_wheel_steer
        step = self.steer_rate * dt
        self.steer += max(-step, min(step, target - self.steer))
        out.steer = self.steer

        # gear: only switch when (nearly) stopped
        want_reverse = v_t < -0.05
        if want_reverse != self.reverse:
            if abs(v_meas) > 0.3:
                out.brake, out.state = 1.0, 'changing direction'
                self.integral = 0.0
                return out
            self.reverse = want_reverse
            self.integral = 0.0
        out.reverse = self.reverse

        # speed along the current gear's direction
        speed, target_speed = (-v_meas, -v_t) if self.reverse else (v_meas, v_t)
        if target_speed < 0.05:
            out.brake = 1.0 if speed > 0.5 else 0.6       # stop, then hold
            self.integral = 0.0
            if out.state == 'tracking':
                out.state = 'stopping' if speed > 0.1 else 'holding'
            return out
        err = target_speed - speed
        self.integral = max(-0.3, min(0.3, self.integral + err * dt))
        u = self.ff0 + self.ff1 * target_speed + self.kp * err + self.ki * self.integral
        if err < -0.5:                                     # well above the target: brake
            out.brake = min(1.0, -0.3 * err)
        elif u > 0.0:
            out.throttle = min(self.max_throttle, u)
        return out
