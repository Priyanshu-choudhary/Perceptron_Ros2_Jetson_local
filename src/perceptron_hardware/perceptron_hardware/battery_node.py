#!/usr/bin/env python3
"""Battery monitor for the Perceptron robot.

Publishes
    /battery/state   sensor_msgs/BatteryState   the full picture
    /battery/power   std_msgs/Float32           watts, see the note below
    /battery/low     std_msgs/Bool              charge below low_battery_percentage
    /battery/critical std_msgs/Bool             nearly flat, or cells sagging below their floor

Right now the values are simulated: constants you can change at runtime. Every
parameter is live, so

    ros2 param set /battery_node voltage 10.2

takes effect on the next publish. That is enough to exercise a "battery low,
go and dock" decision without waiting for a real pack to drain.

WHY THERE IS A SEPARATE POWER TOPIC
sensor_msgs/BatteryState has voltage, current, charge, capacity and percentage,
but no power field. Power is voltage * current, so it is published separately
rather than smuggled into a field that means something else.

CURRENT SIGN
ROS convention: current is NEGATIVE while discharging and positive while
charging. A node that reports +2.5 A on a robot that is driving around is
telling every consumer that the pack is filling up.

STATE OF CHARGE FROM VOLTAGE
LiPo voltage maps to charge through a curve that is famously flat in the middle,
so a linear interpolation between empty and full is wrong by tens of percent
around 50%. The table below is a standard resting-voltage curve. Two caveats
worth knowing:

  * Voltage sags under load. Measure 11.1 V while pulling 8 A and the pack is
    not at the charge that 11.1 V resting would imply. The node compensates
    with a simple ohmic model, V_open = V_measured + |I| * R_internal.
  * The compensation is only as good as internal_resistance, which changes with
    temperature, age and cell chemistry. For anything safety-critical, count
    coulombs instead of trusting voltage.

ON THE REAL ROBOT
stm32_bridge_node already receives bus voltage and current in its 0x55
telemetry frame and currently discards both. Wiring those through and setting
simulate:=false is the intended path; see the package README.
"""

import math

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32

# Resting volts per cell -> state of charge, for LiPo. Descending.
LIPO_CURVE = [
    (4.20, 1.00), (4.15, 0.95), (4.11, 0.90), (4.08, 0.85), (4.02, 0.80),
    (3.98, 0.75), (3.95, 0.70), (3.91, 0.65), (3.87, 0.60), (3.85, 0.55),
    (3.84, 0.50), (3.82, 0.45), (3.80, 0.40), (3.79, 0.35), (3.77, 0.30),
    (3.75, 0.25), (3.73, 0.20), (3.71, 0.15), (3.69, 0.10), (3.61, 0.05),
    (3.27, 0.00),
]


def soc_from_cell_voltage(v: float) -> float:
    """State of charge in 0..1 from a resting per-cell voltage."""
    if v >= LIPO_CURVE[0][0]:
        return 1.0
    if v <= LIPO_CURVE[-1][0]:
        return 0.0
    for (v_hi, s_hi), (v_lo, s_lo) in zip(LIPO_CURVE, LIPO_CURVE[1:]):
        if v_lo <= v <= v_hi:
            span = v_hi - v_lo
            f = 0.0 if span <= 0 else (v - v_lo) / span
            return s_lo + f * (s_hi - s_lo)
    return 0.0


class BatteryNode(Node):

    def __init__(self):
        super().__init__('battery_node')

        # --- pack description ---
        self.declare_parameter('cell_count', 3)                 # 3S
        self.declare_parameter('design_capacity', 5.0)          # Ah
        self.declare_parameter('internal_resistance', 0.05)     # ohm, whole pack
        self.declare_parameter('serial_number', 'SIM-3S-0001')
        self.declare_parameter('location', 'chassis_bay')

        # --- simulated measurements, change these live ---
        self.declare_parameter('simulate', True)
        # Multiplies /battery/measured_current into the ROS convention above.
        # The Perceptron's sensor reads the other way round (+1.1 A driving,
        # -1.3 A on the charger), so battery_real.yaml sets -1.
        self.declare_parameter('measured_current_sign', 1.0)
        self.declare_parameter('voltage', 12.0)                 # V at the terminals
        self.declare_parameter('current', -2.5)                 # A, negative = discharging
        self.declare_parameter('temperature', 25.0)             # degrees C
        self.declare_parameter('present', True)

        # --- optional drain, off by default ---
        self.declare_parameter('simulate_discharge', False)
        self.declare_parameter('discharge_minutes_to_empty', 30.0)

        # --- thresholds ---
        # "Head for the dock" is a question about remaining charge, so it is a
        # percentage. Volts per cell is a bad unit for it: the LiPo curve is
        # nearly flat from 3.7 to 4.0 V and then collapses, so 3.50 V/cell is
        # 3.4% charge rather than the comfortable margin it looks like.
        self.declare_parameter('low_battery_percentage', 0.25)
        self.declare_parameter('critical_battery_percentage', 0.10)
        # Cell protection is a different question and uses the voltage actually
        # at the terminals, sag included, because sag under load is what damages
        # cells. Compensating it away first would let the pack be pulled under
        # its floor while the estimate still looked healthy.
        self.declare_parameter('critical_cell_voltage', 3.30)
        self.declare_parameter('full_cell_voltage', 4.20)
        self.declare_parameter('max_cell_voltage', 4.25)
        self.declare_parameter('min_cell_voltage', 3.00)

        # --- charging while docked ---
        self.declare_parameter('charge_when_docked', True)
        self.declare_parameter('docking_status_topic', '/docking/status')
        self.declare_parameter('charge_current', 2.0)           # A into the pack

        self.declare_parameter('publish_rate', 1.0)             # Hz
        self.declare_parameter('frame_id', 'base_link')

        self._docked = False
        self._sim_voltage = float(self.get_parameter('voltage').value)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.state_pub = self.create_publisher(BatteryState, '/battery/state', 10)
        self.power_pub = self.create_publisher(Float32, '/battery/power', 10)
        self.low_pub = self.create_publisher(Bool, '/battery/low', 10)
        self.crit_pub = self.create_publisher(Bool, '/battery/critical', 10)

        if bool(self.get_parameter('charge_when_docked').value):
            from std_msgs.msg import String
            self.create_subscription(
                String, self.get_parameter('docking_status_topic').value,
                self._docking_cb, 10)

        # Real measurements from stm32_bridge_node's 0x55 frame, used when
        # simulate is false. Kept as the latest value rather than averaged:
        # they arrive at 100 Hz and this node publishes at 1 Hz, so a stale
        # reading is never more than 10 ms old.
        self._measured_voltage = None
        self._measured_current = None
        self.create_subscription(
            Float32, '/battery/measured_voltage', self._voltage_cb, 10)
        self.create_subscription(
            Float32, '/battery/measured_current', self._current_cb, 10)
        self._warned_no_data = False

        rate = max(0.1, float(self.get_parameter('publish_rate').value))
        self._period = 1.0 / rate
        self.create_timer(self._period, self._publish)

        cells = int(self.get_parameter('cell_count').value)
        self.get_logger().info(
            f'Battery node up: {cells}S LiPo, '
            f'{self.get_parameter("design_capacity").value:.1f} Ah, '
            f'simulated={self.get_parameter("simulate").value}. '
            f'Change it live with: ros2 param set /battery_node voltage <V>')

    # ---------------------------------------------------------------- helpers

    def _p(self, name):
        return self.get_parameter(name).value

    def _on_set_parameters(self, params):
        """Accept live changes, and keep the drain simulation in step."""
        for p in params:
            if p.name == 'voltage':
                self._sim_voltage = float(p.value)
        return SetParametersResult(successful=True)

    def _docking_cb(self, msg):
        self._docked = (msg.data.split(' |')[0] == 'DOCKED')

    def _voltage_cb(self, msg):
        self._measured_voltage = float(msg.data)

    def _current_cb(self, msg):
        self._measured_current = float(msg.data)

    # ------------------------------------------------------------------ logic

    def _measure(self):
        """Return (voltage, current): real from the ECU, or simulated."""
        if not bool(self._p('simulate')):
            if self._measured_voltage is None:
                # Do not fall back to the simulated constant here. Reporting a
                # plausible 12.0 V while the ECU is silent is how a flat pack
                # goes unnoticed; say so instead and report the pack absent.
                if not self._warned_no_data:
                    self.get_logger().warn(
                        'simulate=false but no /battery/measured_voltage yet. '
                        'Is stm32_bridge_node running?')
                    self._warned_no_data = True
                return None, None
            if self._warned_no_data:
                self.get_logger().info('battery measurements arrived.')
                self._warned_no_data = False
            current = self._measured_current
            sign = float(self._p('measured_current_sign'))
            return self._measured_voltage, (0.0 if current is None else sign * current)

        cells = int(self._p('cell_count'))
        charging = self._docked and bool(self._p('charge_when_docked'))

        if charging:
            current = float(self._p('charge_current'))
            if bool(self._p('simulate_discharge')):
                full = float(self._p('full_cell_voltage')) * cells
                # Charge back up about four times as fast as the drain, purely
                # so a docking test does not take half an hour.
                minutes = max(0.1, float(self._p('discharge_minutes_to_empty'))) / 4.0
                span = full - float(self._p('min_cell_voltage')) * cells
                self._sim_voltage = min(full, self._sim_voltage
                                        + span * self._period / (minutes * 60.0))
        else:
            current = float(self._p('current'))
            if bool(self._p('simulate_discharge')):
                lo = float(self._p('min_cell_voltage')) * cells
                hi = float(self._p('full_cell_voltage')) * cells
                minutes = max(0.1, float(self._p('discharge_minutes_to_empty')))
                self._sim_voltage = max(lo, self._sim_voltage
                                        - (hi - lo) * self._period / (minutes * 60.0))

        voltage = self._sim_voltage if bool(self._p('simulate_discharge')) \
            else float(self._p('voltage'))
        if charging and not bool(self._p('simulate_discharge')):
            voltage = float(self._p('voltage'))
        return voltage, current

    def _health(self, cell_v, temperature):
        msg = BatteryState
        if cell_v > float(self._p('max_cell_voltage')):
            return msg.POWER_SUPPLY_HEALTH_OVERVOLTAGE
        if cell_v < float(self._p('min_cell_voltage')):
            return msg.POWER_SUPPLY_HEALTH_DEAD
        if temperature > 60.0:
            return msg.POWER_SUPPLY_HEALTH_OVERHEAT
        if temperature < 0.0:
            return msg.POWER_SUPPLY_HEALTH_COLD
        return msg.POWER_SUPPLY_HEALTH_GOOD

    def _status(self, current, percentage):
        msg = BatteryState
        if not bool(self._p('present')):
            return msg.POWER_SUPPLY_STATUS_UNKNOWN
        if current > 0.05:
            return (msg.POWER_SUPPLY_STATUS_FULL if percentage >= 0.99
                    else msg.POWER_SUPPLY_STATUS_CHARGING)
        if current < -0.05:
            return msg.POWER_SUPPLY_STATUS_DISCHARGING
        return msg.POWER_SUPPLY_STATUS_NOT_CHARGING

    # -------------------------------------------------------------- publisher

    def _publish(self):
        cells = int(self._p('cell_count'))
        voltage, current = self._measure()

        if voltage is None:
            # No real measurement yet. Publish an explicitly absent pack rather
            # than a made-up one, so a consumer can tell "not reporting" from
            # "reporting a healthy 12 V".
            msg = BatteryState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = str(self._p('frame_id'))
            msg.voltage = float('nan')
            msg.current = float('nan')
            msg.percentage = float('nan')
            msg.present = False
            msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
            msg.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_UNKNOWN
            msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
            self.state_pub.publish(msg)
            return

        temperature = float(self._p('temperature'))

        # Compensate the load sag before reading the curve, otherwise a healthy
        # pack under acceleration looks nearly flat.
        r_int = float(self._p('internal_resistance'))
        open_circuit = voltage + abs(current) * r_int if current < 0.0 else \
            voltage - abs(current) * r_int
        cell_v = open_circuit / max(1, cells)
        percentage = soc_from_cell_voltage(cell_v)

        design = float(self._p('design_capacity'))

        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self._p('frame_id'))
        msg.voltage = float(voltage)
        msg.current = float(current)
        msg.temperature = float(temperature)
        msg.charge = float(percentage * design)
        msg.capacity = float(design)
        msg.design_capacity = float(design)
        msg.percentage = float(percentage)
        msg.power_supply_status = self._status(current, percentage)
        msg.power_supply_health = self._health(cell_v, temperature)
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
        msg.present = bool(self._p('present'))
        # Even split: there is no per-cell sensing without a BMS. Real balance
        # data would come from the BMS and would not be identical like this.
        msg.cell_voltage = [float(cell_v)] * cells
        msg.cell_temperature = [float('nan')] * cells
        msg.location = str(self._p('location'))
        msg.serial_number = str(self._p('serial_number'))
        self.state_pub.publish(msg)

        self.power_pub.publish(Float32(data=float(voltage * current)))

        # Planning signal: how much is left.
        low = percentage <= float(self._p('low_battery_percentage'))
        # Protection signal: either almost nothing left, or the cells are being
        # dragged below their floor right now. measured_cell_v is deliberately
        # NOT the sag-compensated figure.
        measured_cell_v = voltage / max(1, cells)
        crit = (percentage <= float(self._p('critical_battery_percentage'))
                or measured_cell_v <= float(self._p('critical_cell_voltage')))
        self.low_pub.publish(Bool(data=bool(low)))
        self.crit_pub.publish(Bool(data=bool(crit)))

        if crit:
            self.get_logger().error(
                f'Battery CRITICAL: {percentage * 100:.0f}%, {voltage:.2f} V '
                f'({measured_cell_v:.2f} V/cell measured). Stop and charge.',
                throttle_duration_sec=10.0)
        elif low:
            self.get_logger().warn(
                f'Battery low: {percentage * 100:.0f}%, {voltage:.2f} V '
                f'({cell_v:.2f} V/cell open-circuit). Head for the dock.',
                throttle_duration_sec=30.0)


def main(args=None):
    rclpy.init(args=args)
    node = BatteryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
