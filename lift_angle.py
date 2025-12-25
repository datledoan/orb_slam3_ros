#!/usr/bin/env python3
import rospy
import math
from std_msgs.msg import Float64

class TurntableAngle:
    def __init__(self):
        rospy.init_node("turntable_angle_publisher")

        self.omega = 0.0     # rad/s
        self.theta = 0.0     # rad

        self.last_time = rospy.Time.now()

        rospy.Subscriber(
            "/turntable/angular_velocity",
            Float64,
            self.vel_cb,
            queue_size=1
        )

        self.angle_pub = rospy.Publisher(
            "/turntable/angle",
            Float64,
            queue_size=10
        )

        self.rate = rospy.Rate(100)  # Hz

    def vel_cb(self, msg):
        self.omega = msg.data

    def update(self):
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        if dt <= 0:
            return

        self.theta += self.omega * dt

        # normalize [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        self.last_time = now

        msg = Float64()
        msg.data = self.theta
        self.angle_pub.publish(msg)

    def spin(self):
        while not rospy.is_shutdown():
            self.update()
            self.rate.sleep()


if __name__ == "__main__":
    node = TurntableAngle()
    node.spin()
