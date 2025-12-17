#!/usr/bin/env python3
import rospy
import cv2
import apriltag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class AprilTagOverlayNode:
    def __init__(self):
        rospy.init_node("apriltag_overlay_node")

        self.bridge = CvBridge()

        # Subscriber
        self.sub = rospy.Subscriber(
            "/usb_cam_front",
            Image,
            self.image_callback,
            queue_size=1
        )

        # Publisher
        self.pub = rospy.Publisher(
            "/usb_cam_front/apriltag_overlay",
            Image,
            queue_size=1
        )

        # AprilTag detector
        self.detector = apriltag.Detector()

        rospy.loginfo("AprilTag Overlay Node started")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        detections = self.detector.detect(gray)

        for det in detections:
            tag_id = det.tag_id
            corners = det.corners.astype(int)
            center = det.center.astype(int)

            # Draw bounding box
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Draw center
            cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)

            # Put tag ID
            cv2.putText(
                frame,
                f"ID: {tag_id}",
                (center[0] - 20, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # Publish overlay image
        overlay_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub.publish(overlay_msg)

if __name__ == "__main__":
    try:
        AprilTagOverlayNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
