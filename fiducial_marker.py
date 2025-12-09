#! /usr/bin/env python3

""" Fiducial marker support localization when docking

Publish tf from base_link -> odom_apriltag base on odom topic data
When see a fiducial marker -> publish tf from base_link -> tag -> goal, calibration odom_apriltag coordinate

"""
import os
import cv2
import apriltag
import rospy
import numpy as np
try:
    import  tf_python3 as tf
    from tf_python3.transformations import quaternion_matrix, quaternion_from_matrix, euler_matrix, quaternion_from_euler, euler_from_matrix
except ModuleNotFoundError:
    import tf
    from tf.transformations import quaternion_matrix, quaternion_from_matrix, euler_matrix, quaternion_from_euler, euler_from_matrix
import datetime
from nav_msgs.msg import Odometry
from helper.helper import apritag_pose, apritag_pose_fisheye
from sensor_msgs.msg import CameraInfo, Image
from cros_drop_pallet.srv import Marker, MarkerResponse
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
import yaml
import time
import rospkg
from helper.data_matrix_code_core import DataMatrixCode
from helper import control_function as cf
from geometry_msgs.msg import Twist, Pose
from datetime import datetime

from drop_pallet_server import smooth_velocity
from cros_kinematics.msg import speed_wheel
from std_msgs.msg import UInt8


def pub_tf(pose, frame_cam = "carter_camera_stereo_right", id=0):
    """ give pose of april tag in camera coordinate, publish tf from frame camera to frame tag,
        publish frame tag to goal"""

    # pose = np.linalg.inv(pose)
    

    (trans_bc, rot_bc) = listener.lookupTransform(robot_frame, frame_cam, rospy.Time(0))

    trans = pose[:3, -1]
    rpy = euler_from_matrix(pose[:3,:3])
    quat = quaternion_from_euler(*rpy)
    br.sendTransform(trans,
                    quat,
                    rospy.Time.now(),
                    f"tag_{id}",
                    frame_cam)
    if id == 0:
        br.sendTransform(trans_tag0_goal,
                    quat_tag_goal_id0,
                    rospy.Time.now(),
                    f"goal_{id}",
                    f"tag_{id}")
    elif id != 0:
        br.sendTransform(trans_tag1_goal,
                        quat_tag_goal_id1,
                        rospy.Time.now(),
                        f"goal_{id}",
                        f"tag_{id}")
    
    
    # try:
    #        (trans, rot) = listener.lookupTransform(robot_frame, f"tag_{id}", rospy.Time(0))
    #        print(f"base_link =>tag_{id}", trans, rot)
            
    #        normlize_trans = trans[0], trans[1], 0
    #        normlize_rot = tf.transformations.quaternion_from_euler(0, 0, 0)
    #        norm_mtx_base_tag = quaternion_matrix(normlize_rot)
    #        norm_mtx_base_tag[:3,3] = normlize_trans
    #        norm_mtx_base_cam = norm_mtx_base_tag @ np.linalg.inv(pose)
    #        rpy_base_cam = euler_from_matrix(norm_mtx_base_cam[:3,:3])
    #        trans_base_cam = norm_mtx_base_cam[:3,-1]
    #        print(id, trans_base_cam, rpy_base_cam)
    # except :
    #        pass


def load_coefficients(path, k=1):
    ranges = {7: 720, 12: 1280, 6: 640, 3: 320}
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode("K").mat()
    dist = cv_file.getNode("D").mat()
    w = mtx[0, 2] * 2
    h = mtx[1, 2] * 2
    w = ranges[w // 100]
    h = ranges[h // 100]
    cv_file.release()

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w // k, h // k))
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w // k, h // k), cv2.CV_16SC2
    )

    return newcameramtx, mtx, dist, mapx, mapy, roi

class State:
        def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.v = v
            self.last_update = rospy.get_time()
        def reset(self, x,y,yaw):
            self.x = x
            self.y = y
            self.yaw = yaw
        def pub_odom(self):
            odom_base_mat = euler_matrix(0, 0, self.yaw)
            odom_base_mat[:3,3] = (self.x, self.y, 0)
            base_odo_mat = np.linalg.inv(odom_base_mat)

            br.sendTransform(base_odo_mat[:3,-1],
                    quaternion_from_matrix(base_odo_mat),
                    rospy.Time.now(),
                    fiducial_marker_param["odom_frame"],
                    robot_frame)
       

class RosInterface:
        def __init__(self):
            self.node_name = rospy.get_name()
            self.topic_rgb_front = fiducial_marker_param["camera_topic"]
            if fiducial_marker_param['simulation']:
                camera_info = rospy.wait_for_message("/camera_info_right", CameraInfo, timeout=5)
                self.newcameramatrix = np.array(camera_info.K).reshape(3,3)
            else:
                fcam = fiducial_marker_param["camera_coef_file"]
                self.newcameramatrix, self.matrix_coefficients, self.distortion_coefficients, self.mapx, self.mapy, roi = load_coefficients(f"{module_path}/scripts/camera_info/{fcam}")
            camera_params = [self.newcameramatrix[0,0], self.newcameramatrix[1,1], self.newcameramatrix[0,2], self.newcameramatrix[1,2]]
            option  = apriltag.DetectorOptions(refine_pose=True)
            self.detector = apriltag.Detector(option)
            self.tag_size = fiducial_marker_param["tag_size"]
            self.state = State()
            self.rgb_msg_front = None
            self.overlay  = np.zeros((500, 500, 3))
            self.d435_msg = None
            self.enable = False
            self.available_tag = False
            self.emergency_state = 1

            rospy.Service(self.node_name + '/get_id', Marker, self.get_id_srv)
            rospy.Service(self.node_name + '/init_pose', Trigger, self.reset_odom_srv)
            rospy.Service(self.node_name + '/enable', SetBool, self.enable_srv)
            rospy.Subscriber('odom', Odometry, self.odom_callback, queue_size=1)
            rospy.Subscriber(self.topic_rgb_front, Image, self.callback_rgb_front)
            rospy.Subscriber("/fw_state", UInt8, self.emergency_cb) # check emergency state

            self.aux_tag = 0
            self.in_action = False

            self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
            self.pub_speed = rospy.Publisher('/wheel_speed_cmd', speed_wheel, queue_size=1)
            rospy.sleep(0.5)
            self.mat_base_cam = self.get_mat_base_cam()  

        def emergency_cb(self, data):
            self.emergency_state = data.data         

        def get_id_srv(self, req):
            try:
                data = rospy.wait_for_message(req.topic,Image, timeout=3)
            except rospy.exceptions.ROSException as e:
                rospy.logwarn(f"[fiducial marker]: {e}")
                return MarkerResponse([], Pose(), "No topic data")
            image = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, -1))[:, :, ::-1]
            #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_u = image
	
            
            # frame_u  = cv2.undistort(image, self.matrix_coefficients, self.distortion_coefficients, None, self.newcameramatrix)
            gray = cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY)
            detections, dimg = self.detector.detect(gray, return_image = True)
            overlay = frame_u // 2 + dimg[:, :, None] // 2
            for i, detection in enumerate(detections):
                tag_id = detection.tag_id
                cv2.putText(overlay, str(tag_id), (int(detection.center[0]),int(detection.center[1])), cv2.FONT_HERSHEY_PLAIN, 3, (255,244,0), 2)


            #TODO: remove image if full memory
            time_string = datetime.now().strftime('%Y/%m/%d')
            posfix_name = datetime.now().strftime('%H:%M:%S')
            camId = req.topic.replace('/image_raw',"")
            folder = f"{module_path}/scripts/results/{time_string}/{req.robotId}"
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(f"{folder}/{req.orderId}_action_{camId}_{posfix_name}.png", overlay)
            detections.sort(key=lambda det: det.center[1])  # sort by y axis


            list_id = [det.tag_id for det in detections]
            res = MarkerResponse(list_id, Pose(), "succees")
            return res


        def enable_srv(self, req):
            self.enable = req.data
            if not self.enable:
                self.available_tag = False
            rospy.loginfo(f"[{self.node_name}] enable: {self.enable}")
            rospy.sleep(1)
            if self.rgb_msg_front is None:
                return SetBoolResponse(False, "Camera Error!")
            else:
                return SetBoolResponse(True, "Successful!")
        
        def get_mat_base_cam(self):
            try:
                trans, rot = listener.lookupTransform(robot_frame, camera_frame, rospy.Time(0))
                mat_base_cam = quaternion_matrix(rot)
                mat_base_cam[:3,-1] = trans
            except tf.LookupException:
                rospy.logerr(f"[Fiducial marker]: cannot lookup transform from {robot_frame} to {camera_frame}")
                mat_base_cam = np.eye(4)
            return mat_base_cam

        def set_speed_wheel(self, vel_x, vel_yaw):
            sp_w = -int(vel_yaw * controller_param['ratio_rad_wheel'])
            sp_upper  = int(-sp_w * controller_param['ratio_wheel_w_upper'])
            sp_pub = speed_wheel(sp_w, -sp_w, 0, sp_upper)
            self.pub_speed.publish(sp_pub)

        def set_cmd_vel(self, vel_x, vel_yaw):
            twist = Twist()
            twist.linear.x = vel_x
            twist.angular.z = vel_yaw
            self.cmd_pub.publish(twist)
        def reset_odom_srv(self, req):
            self.aux_tag = 0
            self.in_action = False
            rospy.sleep(2)
            rospy.loginfo('[fiducial_marker] Aux tag id:'+ str(self.aux_tag))
            self.in_action = True
            if self.aux_tag == 0:
                if self.rgb_msg_front is not None:
                    cv2.imwrite("/home/aaeon/vmr/ws_develop/src/services/cros_drop_pallet/scripts/results/debug.png", self.rgb_msg_front[0])
                rospy.logerr("Not found aux tag, there should be 2 april tag in front of robot")
                return TriggerResponse(message = "Not found aux tag, there should be 2 april tag in front of robot", success=False)
            res, msg = self.get_apriltag_pose(tag_id=self.aux_tag)
            if res is None:
                return TriggerResponse(message = msg, success=False)
            else:
                self.state.reset(*res)
                if abs(res[1]) >0.05: #distance vertical thresh
                    rospy.loginfo("[Fiducial marker]: Robot is too far destination, trying setup position")
                    self.prepare_position(res)
                return TriggerResponse(message = msg, success=True)

        def move_forward(self, dis):
            v_ref = controller_param['v_ref']/2
            current_x = self.state.x
            current_y = self.state.y

            while not rospy.is_shutdown():
                if self.emergency_state !=1 and self.emergency_state != 12:
                    rospy.loginfo(f"[fiducial_marker]: Stop move_forward Emergency state = {self.emergency_state}") 
                    break
                distance_move = np.hypot(self.state.x-current_x, self.state.y-current_y)
                error = abs(dis) - distance_move
                if abs(error) < controller_param['max_distance']:
                    self.set_cmd_vel(0, 0)
                    break
                v = v_ref * smooth_velocity(abs(error), 0.5)
                self.set_cmd_vel(v, 0)
                rospy.sleep(0.1)

        def rotate(self, target_rotate):
            # rospy.loginfo(f"[fiducial_marker]: Rotating target {target_rotate:.2f}, current: {self.state.yaw:.2f}") 
            w_ref = controller_param['w_ref']

            while not rospy.is_shutdown():
                qr_yaw = target_rotate - self.state.yaw
                if self.emergency_state !=1 and self.emergency_state != 12:
                    rospy.loginfo(f"[fiducial_marker]: Stop rotating Emergency state = {self.emergency_state}") 
                    break
                if abs(qr_yaw) < controller_param['max_angle']:
                    self.set_speed_wheel(0, 0)
                    break
                w = 2*w_ref * smooth_velocity(abs(qr_yaw), 2*np.pi/3) * np.sign(qr_yaw)
                # print(qr_yaw, w)
                self.set_speed_wheel(0, w)
                rospy.sleep(0.01)

        def prepare_position(self, current_pose):
            x,y,yaw = current_pose
            if y > 0:
                target_rotate = self.state.yaw + -yaw - np.pi/2
            else:
                target_rotate =  self.state.yaw + -yaw + np.pi/2
            self.rotate(target_rotate)
            rospy.sleep(0.1)
            self.move_forward(y)
            rospy.sleep(0.1)
            self.rotate(self.state.yaw + np.pi/2 * np.sign(y))
            


        def odom_callback(self, data):
            linear = data.twist.twist.linear.x
            angular_z = data.twist.twist.angular.z
            current_time = rospy.get_time()
            vel_dt_ = current_time - self.state.last_update
            self.state.last_update = current_time
            delta_linear = linear * vel_dt_
            delta_z = angular_z * vel_dt_
            self.state.yaw += delta_z
            self.state.x += delta_linear * np.cos(self.state.yaw)
            self.state.y += delta_linear * np.sin(self.state.yaw)
            self.v = linear
            # startt = time.time()
        #   
            # print(time.time()-startt)

        def callback_rgb_front(self, data_msg):
            rgb_msg_front = np.frombuffer(data_msg.data, dtype=np.uint8).reshape(
            (data_msg.height, data_msg.width, -1))[:, :, ::-1]
            self.rgb_msg_front = [rgb_msg_front, data_msg.header.stamp]

        def get_apriltag_pose(self, tag_id):
            # cv2.imwrite("/home/aaeon/vmr/cuong_ws/src/aprilt.png", self.overlay)

            if not self.available_tag:
                rospy.loginfo("[fiducial_marker] Currently, not see any tag")
                return None, "Cannot"
            try:
                (trans, rot) = listener.lookupTransform(f"goal_{tag_id}", robot_frame , rospy.Time(0))
                # print("trans, rot", f"goal_{tag_id}", robot_frame, trans, rot )
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(f"[fiducial_marker]: {e}")
                return None, "Cannot get tf"
            rpy = euler_from_matrix(quaternion_matrix(rot))
            x,y = trans[:2]
            yaw = rpy[2]
            rospy.loginfo(f"[fiducial_marker] Nearest tag: id={tag_id}, x = {x:.3f}, y = {y:.3f}, yaw={yaw:.3f}")
            return (x,y,yaw), "success"
        def update_odom_from_goal(self, tag_id, mat_cam_goal):
            mat_base_goal = self.mat_base_cam @ mat_cam_goal
            mat_goal_base = np.linalg.inv(mat_base_goal)
            rpy = euler_from_matrix(mat_goal_base[:3,:3])
            # print(datetime.datetime.now().strftime("%H:%M:%S"), f"update odom_apriltal id {tag_id}")
            self.state.reset(*mat_goal_base[:2,-1], rpy[2])
    
        def loop(self):
            while not rospy.is_shutdown():
                updated = False
                if self.rgb_msg_front is None or not self.enable:
                    rospy.sleep(0.5)
                    continue
                frame, stamp = self.rgb_msg_front
                ## frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # if fiducial_marker_param["simulation"]:
                #     frame_u = frame
                # else:
                    # frame_u  = cv2.undistort(frame, self.matrix_coefficients, self.distortion_coefficients, None, self.newcameramatrix)
                    # frame_u = cv2.remap(frame, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR)
                # gray = cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections, dimg = self.detector.detect(gray, return_image = True)
                # overlay = frame_u // 2 + dimg[:, :, None] // 2
                overlay = frame // 2 + dimg[:, :, None] // 2
                self.overlay = overlay
                sorted_detections = sorted(detections, key=lambda x: x.tag_id)
                self.available_tag = len(detections) > 0
                if not updated:
                    for i, detection in enumerate(sorted_detections):
                        tag_id = detection.tag_id
                        # pose2,rvec,tvec = apritag_pose(detection.corners[::-1], fiducial_marker_param['tag_size'], self.newcameramatrix)
                        pose2,rvec,tvec = apritag_pose_fisheye(detection.corners[::-1], fiducial_marker_param['tag_size'], self.newcameramatrix, self.distortion_coefficients)
                        
                        # apriltag._draw_pose(overlay, camera_params, fiducial_marker_param['tag_size'], pose2)
                        if self.in_action and tag_id!= self.aux_tag and tag_id!=0:
                            continue

                        if tag_id != 0: # only update nearest tag
                            self.aux_tag = tag_id
                            updated = True
                            pub_tf(pose2,camera_frame, id=tag_id)
                            mat_cam_goal = pose2 @ mat_tag_goal_id1
                            self.update_odom_from_goal(tag_id, mat_cam_goal)
                        
                        elif tag_id == 0  and not updated:
                            mat_cam_goal = pose2 @ mat_tag_goal_id0
                            pub_tf(pose2,camera_frame, id=tag_id)

                            self.update_odom_from_goal(tag_id, mat_cam_goal)

                    
                    
                        

                # cv2.imshow("aaa", overlay)
                # cv2.waitKey(1)
                self.state.pub_odom()
                # rospy.sleep(0.02)


def read_cfg_tag_goal(gtag):
    trans_goal_tag = [gtag['x'], gtag['y'], 0]
    quat_goal_tag = quaternion_from_euler(0, 0, np.radians(gtag['yaw']))
    mat_goal_tag = quaternion_matrix(quat_goal_tag)
    mat_goal_tag[:3, -1] = trans_goal_tag
    mat_tag_goal = np.linalg.inv(mat_goal_tag)
    trans_tag_goal = mat_tag_goal[:3, -1]
    quat_tag_goal = quaternion_from_matrix(mat_tag_goal)
    return mat_tag_goal, quat_tag_goal, trans_tag_goal


if __name__ == "__main__":
    module_path = rospkg.RosPack().get_path("cros_drop_pallet")

    with open(module_path+"/config/drop_pallet.yaml", "r") as f:
        controller_param = yaml.safe_load(f)

    with open(module_path+"/config/fiducial_marker.yaml", "r") as f:
        fiducial_marker_param = yaml.safe_load(f)

    robot_frame = fiducial_marker_param["robot_frame"]
    camera_frame = fiducial_marker_param["camera_frame"]



    mat_tag_goal_id1, quat_tag_goal_id1, trans_tag1_goal = read_cfg_tag_goal(fiducial_marker_param['goal_to_tag_1'])
    mat_tag_goal_id0, quat_tag_goal_id0, trans_tag0_goal = read_cfg_tag_goal(fiducial_marker_param['goal_to_tag_0'])


    
    # rospy.set_param("/use_sim_time", True)

    rospy.init_node("fiducial_marker")
    listener = tf.TransformListener()
    br = tf.TransformBroadcaster()
    
    
    rf = RosInterface()
    rf.loop()
    rospy.spin()

    
    
