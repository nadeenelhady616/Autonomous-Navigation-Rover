#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan


class PID:
    """Small standalone PID so tuning logic isn't tangled up with ROS
    plumbing. setpoint is always 0 here (we want center_error_px -> 0)."""

    def __init__(self, kp, ki, kd, output_limit, integral_limit, derivative_filter_alpha):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Clamp final output (e.g. max angular.z in rad/s)
        self.output_limit = output_limit

        # Clamp the integral term itself (anti-windup) - without this,
        # if the lane is lost or error stays large for a while, the
        # integral term keeps growing and causes a violent overcorrection
        # once the error finally goes away.
        self.integral_limit = integral_limit

        # Derivative term is computed on a low-pass-filtered error signal.
        # Raw pixel error is noisy frame-to-frame; differentiating noise
        # directly amplifies it and causes jittery steering. 0 = no
        # filtering (raw), closer to 1 = heavier smoothing (more lag).
        self.derivative_filter_alpha = derivative_filter_alpha

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_filtered_error = 0.0
        self._initialized = False

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_filtered_error = 0.0
        self._initialized = False

    def update(self, error, dt):
        if dt <= 0.0:
            return 0.0

        if not self._initialized:
            # Avoid a derivative spike on the very first sample
            self._prev_filtered_error = error
            self._prev_error = error
            self._initialized = True

        # Low-pass filter the error before differentiating
        alpha = self.derivative_filter_alpha
        filtered_error = alpha * self._prev_filtered_error + (1 - alpha) * error

        proportional = error

        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

        derivative = (filtered_error - self._prev_filtered_error) / dt

        output = (self.kp * proportional) + (self.ki * self._integral) + (self.kd * derivative)
        output = max(-self.output_limit, min(self.output_limit, output))

        self._prev_error = error
        self._prev_filtered_error = filtered_error

        return output


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class LaneControllerNode(Node):
    def __init__(self):
        super().__init__("lane_controller_node")

        # ================= Parameters =================

        # Must match lane_detector_node's frame_width - used to normalize
        # pixel error into a roughly resolution-independent [-1, 1] range
        # so PID gains don't need retuning if you change camera resolution.
        self.frame_width = 320

        # ---- Speed ----
        # Forward speed in m/s (or whatever units your /cmd_vel consumer
        # expects). Keep this modest until the steering is tuned - a
        # slower robot is much more forgiving of an untuned PID.
        self.linear_speed = 0.15

        # If True, forward speed is reduced when the steering error is
        # large (sharp correction needed) and restored to full speed
        # when error is near zero. Helps stability on sharp turns.
        # Set False for constant speed regardless of error.
        self.reduce_speed_on_error = True
        self.min_speed_fraction = 0.5  # slowest allowed, as a fraction of linear_speed

        # ---- PID gains (this is what you'll actually tune) ----
        # error is normalized to roughly [-1, 1] (see frame_width above),
        # so these gains are in "output per normalized error unit", not
        # "output per pixel." Output itself is angular.z in rad/s.
        #
        # kp : proportional gain. Bigger error -> bigger correction.
        #   - Raise if the robot is slow/sluggish to correct back to center.
        #   - Lower if the robot oscillates side to side (overshoots,
        #     corrects too hard, overshoots the other way).
        #
        # ki : integral gain. Corrects small steady-state bias (e.g. robot
        #   consistently drifts slightly left even at "zero" error).
        #   - Start at 0. Only raise this if you see a persistent small
        #     offset that kp alone never fully removes.
        #   - Too high -> slow oscillation / overshoot that builds up
        #     over time (windup) - if you see this, lower ki or tighten
        #     integral_limit below.
        #
        # kd : derivative gain. Damps oscillation/overshoot by reacting
        #   to how fast the error is changing, not just its size.
        #   - Raise if kp is high enough to respond well but the robot
        #     overshoots/oscillates around center.
        #   - Too high -> twitchy, jittery steering (amplifying noise) -
        #     if you see this, lower kd or increase derivative_filter_alpha.
        self.kp = 0.9
        self.ki = 0.0
        self.kd = 0.15

        # Max magnitude of angular.z the controller is allowed to command
        self.max_angular_z = 1.0

        # Anti-windup clamp on the integral term (in the same normalized
        # error*time units) - keep this small relative to max_angular_z
        self.integral_limit = 0.5

        # Derivative smoothing: 0 = no smoothing (raw, jittery), 0.8 =
        # heavy smoothing (smooth, but laggier response). Start around 0.6.
        self.derivative_filter_alpha = 0.6

        # Control loop rate - decoupled from camera frame rate so steering
        # stays smooth even if lane_detector_node is only publishing every
        # Nth frame. Uses the latest known error/detected state each tick.
        self.control_rate_hz = 20.0

        self._pid = PID(
            self.kp,
            self.ki,
            self.kd,
            self.max_angular_z,
            self.integral_limit,
            self.derivative_filter_alpha,
        )

        self._latest_error_px = 0.0
        self._lane_detected = False
        self._last_tick_time = time.monotonic()

        # ---- Lidar obstacle avoidance (simple reactive layer) ----
        # This sits ON TOP of lane-following: it never re-plans a path,
        # it just watches three forward-facing sectors of /scan and
        # overrides/biases the lane PID's output when something physical
        # is too close. Kept deliberately simple - no costmap, no nav.
        self.scan_topic = "/scan"

        # Forward "danger" cone, measured from straight ahead (0 rad),
        # +/- this many degrees on each side. Only obstacles inside this
        # cone count as "in front of the robot".
        self.front_half_angle_deg = 25.0

        # Left/right "which way is more open" comparison sectors, just
        # outside the front cone above, out to this many degrees. Used
        # only to decide WHICH WAY to turn away from an obstacle.
        self.side_half_angle_deg = 75.0

        # Inside this distance (m), directly ahead: full emergency stop.
        # Robot halts forward motion and rotates in place toward the
        # side with more room until the front cone clears.
        self.obstacle_stop_distance = 0.35

        # Inside this distance (m) but outside stop_distance: not an
        # emergency yet, but close enough to start biasing steering away
        # from the obstacle and slowing down, on top of normal lane
        # following. No effect at/beyond this distance.
        self.obstacle_slow_distance = 0.8

        # Angular speed (rad/s) used both for the in-place emergency turn
        # and as the max magnitude of the blended-in avoidance bias in
        # the slow zone.
        self.obstacle_turn_speed = 0.6

        # How much forward speed is cut (as a fraction, 0-1) at maximum
        # intrusion into the slow zone (i.e. right at stop_distance,
        # just before the emergency stop kicks in).
        self.obstacle_max_slowdown = 0.6

        self._latest_scan = None

        # ================= ROS =================
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.error_sub = self.create_subscription(
            Float32,
            "/lane/center_error_px",
            self.error_callback,
            qos,
        )

        self.detected_sub = self.create_subscription(
            Bool,
            "/lane/detected",
            self.detected_callback,
            qos,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos,
        )

        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)

        self.control_timer = self.create_timer(
            1.0 / self.control_rate_hz,
            self.control_loop,
        )

        self.get_logger().info("Lane controller node started (with lidar obstacle avoidance)")

    def error_callback(self, msg: Float32):
        self._latest_error_px = msg.data

    def detected_callback(self, msg: Bool):
        was_detected = self._lane_detected
        self._lane_detected = msg.data

        # Reset PID state on the transition back to "detected" so a
        # stale integral/derivative from before the loss doesn't cause
        # a jerky correction the moment the lane reappears.
        if self._lane_detected and not was_detected:
            self._pid.reset()

    def scan_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def _sector_min_range(self, scan: LaserScan, angle_lo, angle_hi):
        """Minimum valid range (m) among beams whose angle falls within
        [angle_lo, angle_hi] radians, measured from straight ahead (0 = the
        robot's forward +x axis). Returns None if no valid beam falls in
        that window (e.g. all inf/out-of-range, or window off the scan's
        actual coverage)."""
        if scan is None or not scan.ranges:
            return None

        best = None
        angle = scan.angle_min
        for r in scan.ranges:
            # Normalize into (-pi, pi] so the front cone (which straddles
            # 0) compares correctly regardless of how the driver reports
            # angle_min/angle_max.
            a = math.atan2(math.sin(angle), math.cos(angle))

            if angle_lo <= a <= angle_hi:
                if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                    if best is None or r < best:
                        best = r

            angle += scan.angle_increment

        return best

    def _get_scan_sectors(self):
        """Returns (front_dist, left_dist, right_dist) in meters. Missing
        scan data or no valid return in a sector reads as 'clear' (inf) so
        the robot fails open into normal lane-following rather than
        freezing if the lidar hiccups."""
        scan = self._latest_scan
        front_half = math.radians(self.front_half_angle_deg)
        side_half = math.radians(self.side_half_angle_deg)

        front = self._sector_min_range(scan, -front_half, front_half)
        left = self._sector_min_range(scan, front_half, side_half)
        right = self._sector_min_range(scan, -side_half, -front_half)

        front = front if front is not None else float("inf")
        left = left if left is not None else float("inf")
        right = right if right is not None else float("inf")

        return front, left, right

    def control_loop(self):
        now = time.monotonic()
        dt = now - self._last_tick_time
        self._last_tick_time = now

        # ROS 2 Jazzy's diff_drive_controller requires cmd_vel to be
        # TwistStamped (plain Twist is a different message type and will
        # never connect to the controller's subscriber - no error, it just
        # silently does nothing).
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = self.get_clock().now().to_msg()
        twist_stamped.header.frame_id = "base_footprint"

        front_dist, left_dist, right_dist = self._get_scan_sectors()

        if front_dist < self.obstacle_stop_distance:
            # ---- Emergency: something is right in front of us. ----
            # Stop advancing and rotate in place toward whichever side
            # currently has more room. Reset the lane PID so it doesn't
            # resume with stale integral/derivative once we pull away and
            # go back to lane-following.
            linear_x = 0.0
            angular_z = self.obstacle_turn_speed if left_dist >= right_dist else -self.obstacle_turn_speed
            self._pid.reset()

        else:
            # ---- Baseline: lane-following (unchanged behavior) ----
            if self._lane_detected:
                normalized_error = self._latest_error_px / (self.frame_width / 2.0)

                # Positive center_error_px means the detected lane center is
                # to the right of image center, i.e. the robot needs to turn
                # right to recenter. In ROS REP103, positive angular.z is a
                # LEFT (counter-clockwise) turn, so the correction is negated.
                angular_z = -self._pid.update(normalized_error, dt)

                if self.reduce_speed_on_error:
                    error_fraction = min(1.0, abs(normalized_error))
                    speed_fraction = 1.0 - error_fraction * (1.0 - self.min_speed_fraction)
                    linear_x = self.linear_speed * speed_fraction
                else:
                    linear_x = self.linear_speed

            else:
                # Lane not detected: drive straight, no steering correction,
                # until it's found again. PID state gets reset on
                # reacquisition in detected_callback above, so it doesn't
                # accumulate error while blind.
                angular_z = 0.0
                linear_x = self.linear_speed

            # ---- Lidar slow-zone bias, layered on top of the above ----
            # Not close enough for the emergency stop, but close enough to
            # start nudging away from the obstacle and easing off speed.
            # intrusion goes 0 (at slow_distance) -> 1 (at stop_distance).
            if front_dist < self.obstacle_slow_distance:
                span = self.obstacle_slow_distance - self.obstacle_stop_distance
                if span > 0:
                    intrusion = (self.obstacle_slow_distance - front_dist) / span
                else:
                    intrusion = 1.0
                intrusion = _clamp(intrusion, 0.0, 1.0)

                avoid_bias = self.obstacle_turn_speed * intrusion
                angular_z += avoid_bias if left_dist >= right_dist else -avoid_bias
                angular_z = _clamp(angular_z, -self.max_angular_z, self.max_angular_z)

                linear_x *= (1.0 - self.obstacle_max_slowdown * intrusion)

        twist_stamped.twist.linear.x = linear_x
        twist_stamped.twist.angular.z = angular_z

        self.cmd_pub.publish(twist_stamped)


def main():
    rclpy.init()
    node = LaneControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Publish a zero-velocity command on shutdown so the robot
        # doesn't keep driving with the last command if this node dies.
        try:
            stop_twist = TwistStamped()
            stop_twist.header.stamp = node.get_clock().now().to_msg()
            stop_twist.header.frame_id = "base_footprint"
            node.cmd_pub.publish(stop_twist)
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()