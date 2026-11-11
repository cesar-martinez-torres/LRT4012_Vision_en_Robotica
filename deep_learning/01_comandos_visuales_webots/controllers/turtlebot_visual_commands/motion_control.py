"""Feedback control in metres and radians for the TurtleBot3 Burger."""

import math

WHEEL_RADIUS = 0.033
AXLE_LENGTH = 0.160


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class MotionControl:
    def __init__(self, distance=0.42, turn_degrees=90.0):
        self.distance = distance
        self.turn = math.radians(turn_degrees)
        self.heading = 0.0

    def start(self, command, left, right):
        self.command = command
        self.left_start, self.right_start = left, right
        self.target_distance = self.distance if command == "forward" else 0.0
        if command in ("left", "right"):
            self.heading = wrap_angle(
                self.heading + (self.turn if command == "left" else -self.turn)
            )

    def update(self, left, right, yaw):
        travel = WHEEL_RADIUS * (
            left - self.left_start + right - self.right_start
        ) / 2.0
        distance_error = self.target_distance - travel
        heading_error = wrap_angle(self.heading - yaw)
        distance_ok = abs(distance_error) <= 0.0005
        heading_ok = abs(heading_error) <= math.radians(0.5)
        done = (distance_ok and heading_ok) if self.command == "forward" else heading_ok
        if self.command == "stop" or done:
            return 0.0, 0.0, True, distance_error, heading_error
        # Slow down near the target; hold absolute heading while advancing.
        linear = max(-0.10, min(0.10, 1.8 * distance_error)) if self.command == "forward" else 0.0
        angular = max(-0.8, min(0.8, 4.0 * heading_error))
        left_speed = (linear - AXLE_LENGTH * angular / 2.0) / WHEEL_RADIUS
        right_speed = (linear + AXLE_LENGTH * angular / 2.0) / WHEEL_RADIUS
        return left_speed, right_speed, False, distance_error, heading_error
