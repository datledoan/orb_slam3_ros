#!/usr/bin/env python3
import rospy
import math
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import tf

class DiffDriveOdom:
    def __init__(self):
        rospy.init_node("diff_drive_odom")

        self.R = rospy.get_param("~wheel_radius", 0.05)      # m
        self.L = rospy.get_param("~wheel_separation", 0.30)  # m

        self.w_l = 0.0
        self.w_r = 0.0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.last_time = rospy.Time.now()

        rospy.Subscriber("/wheel_left_velocity", Float64, self.left_cb)
        rospy.Subscriber("/wheel_right_velocity", Float64, self.right_cb)

        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)

        self.rate = rospy.Rate(50)  # Hz

    def left_cb(self, msg):
        self.w_l = msg.data

    def right_cb(self, msg):
        self.w_r = msg.data

    def update(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        if dt <= 0:
            return

        v = self.R * 0.5 * (self.w_r + self.w_l)
        w = self.R / self.L * (self.w_r - self.w_l)

        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw += w * dt

        self.last_time = current_time

        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        q = tf.transformations.quaternion_from_euler(0, 0, self.yaw)
        odom.pose.pose.orientation = Quaternion(*q)

        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w

        self.odom_pub.publish(odom)

    def spin(self):
        while not rospy.is_shutdown():
            self.update()
            self.rate.sleep()


if __name__ == "__main__":
    node = DiffDriveOdom()
    node.spin()
