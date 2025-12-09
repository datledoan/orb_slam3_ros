import numpy as np
import cv2, math
def draw_pose(overlay, camera_params, tag_size, pose, z_sign=1):
    opoints = np.array([
        -1, -1, 0,
        1, -1, 0,
        1, 1, 0,
        -1, 1, 0,
        -1, -1, -2 * z_sign,
        1, -1, -2 * z_sign,
        1, 1, -2 * z_sign,
        -1, 1, -2 * z_sign,
    ]).reshape(-1, 1, 3) * 0.5 * tag_size

    edges = np.array([
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        0, 4,
        1, 5,
        2, 6,
        3, 7,
        4, 5,
        5, 6,
        6, 7,
        7, 4
    ]).reshape(-1, 2)

    fx, fy, cx, cy = camera_params

    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)

    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]

    dcoeffs = np.zeros(5)

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)

    ipoints = np.round(ipoints).astype(int)

    ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

    for i, j in edges:
        cv2.line(overlay, ipoints[i], ipoints[j], (255, 255, 0), 1, 16)
def apritag_pose(corners, tag_size, camera_matrix):
    obj_pts = np.array([[-tag_size / 2, -tag_size / 2, 0],
                        [tag_size / 2, -tag_size / 2, 0],
                        [tag_size / 2, tag_size / 2, 0],
                        [-tag_size / 2, tag_size / 2, 0],
                        ])
    retval, rvec, tvec = cv2.solvePnP(obj_pts, np.asarray(corners, dtype=np.float32),
                                      camera_matrix, np.zeros(5))
    translate_matrix = np.eye(4)
    translate_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
    translate_matrix[:3, -1] = tvec.reshape(3)

    return translate_matrix, rvec, tvec

def apritag_pose_fisheye(corners, tag_size, K, D):
    obj_pts = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)

    pts = corners.reshape(-1, 1, 2).astype(np.float32)

    undistorted = cv2.undistortPoints(pts, K, D, P=K)
    undistorted = undistorted.reshape(-1, 2)

    retval, rvec, tvec = cv2.solvePnP(
        obj_pts,
        undistorted,
        K,
        np.zeros(5)
    )

    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec[:, 0]

    return T, rvec, tvec

def bundle_apritag_pose(corners, tag_size, camera_matrix):
    obj_pts = np.array([[-tag_size / 2, -tag_size / 2, 0],
                        [tag_size / 2, -tag_size / 2, 0],
                        [tag_size / 2, tag_size / 2, 0],
                        [-tag_size / 2, tag_size / 2, 0],
                        ])
    retval, rvec, tvec = cv2.solvePnP(obj_pts, np.asarray(corners, dtype=np.float32),
                                      camera_matrix, np.zeros(5))
    translate_matrix = np.eye(4)
    translate_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
    translate_matrix[:3, -1] = tvec.reshape(3)

    return translate_matrix, rvec, tvec

