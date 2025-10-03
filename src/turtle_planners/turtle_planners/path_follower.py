#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
import math

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')

        # Suscripciones
        self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)

        # Publicador de velocidades
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Variables
        self.current_path = []
        self.current_pose = None
        self.goal_tolerance = 0.15
        self.linear_speed = 0.15
        self.angular_speed = 0.5
        self.target_idx = 0

        self.get_logger().info("✅ PathFollower inicializado")

    def path_callback(self, msg: Path):
        self.current_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.target_idx = 0
        if self.current_path:
            self.get_logger().info(f"Nuevo path recibido con {len(self.current_path)} puntos")

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose
        self.follow_path()

    def follow_path(self):
        if not self.current_path or self.current_pose is None:
            return

        if self.target_idx >= len(self.current_path):
            self.get_logger().info("🎯 Path completado")
            self.stop_robot()
            return

        # Pose actual
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        q = self.current_pose.orientation
        yaw = self.quaternion_to_yaw(q)

        # Siguiente punto objetivo
        tx, ty = self.current_path[self.target_idx]

        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - yaw)

        cmd = Twist()

        if dist < self.goal_tolerance:
            self.get_logger().info(f"✅ Punto {self.target_idx} alcanzado")
            self.target_idx += 1
            self.stop_robot()
            return

        # Control simple (primero girar hacia el objetivo, luego avanzar)
        if abs(angle_error) > 0.3:  # necesita girar bastante
            cmd.angular.z = self.angular_speed * (1.0 if angle_error > 0 else -1.0)
        else:
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 0.5 * angle_error

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
