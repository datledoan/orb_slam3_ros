#!/usr/bin/env python3
import cv2
import apriltag
import numpy as np
import yaml


def load_camera_calib(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = cv_file.getNode("K").mat()
    D = cv_file.getNode("D").mat()
    cv_file.release()

    print("[INFO] Loaded K =\n", K)
    print("[INFO] Loaded D =\n", D)

    return K, D


def apritag_pose_fisheye(corners, tag_size, K, D):
    """
    corners: (4,2) pixel coordinates (fisheye image)
    """

    # Define 3D tag corners in tag coordinate
    obj_pts = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)

    # Convert pixel -> normalized via fisheye model
    pts = corners.reshape(-1, 1, 2).astype(np.float32)

    undistorted = cv2.undistortPoints(pts, K, D, P=K)
    undistorted = undistorted.reshape(-1, 2)

    # SolvePnP with normalized non-distorted points
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        undistorted,
        K,
        np.zeros(5)  # NO distortion here
    )

    if not ok:
        return None, None, None

    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec[:, 0]

    return T, rvec, tvec

def convert_pose_new_to_old(T_new):
    """
    Convert the fisheye pose (new) into the same coordinate convention
    as the old undistorted AprilTag pipeline.
    """
    # Flip Y and Z axes to match old corner-flip convention
    F = np.diag([1, -1, -1])  

    R_new = T_new[:3, :3]
    t_new = T_new[:3, 3]

    R_old = R_new @ F

    T_old = np.eye(4)
    T_old[:3, :3] = R_old
    T_old[:3, 3] = t_new

    return T_old

def main():
    # -------------------------------------------------------
    # 1. Load config
    # -------------------------------------------------------
    with open("fiducial_marker.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    tag_size = cfg["tag_size"]
    camera_coef_file = cfg["camera_coef_file"]

    # -------------------------------------------------------
    # 2. Load fisheye calibration
    # -------------------------------------------------------
    K, D = load_camera_calib(camera_coef_file)

    # -------------------------------------------------------
    # 3. Init AprilTag detector
    # -------------------------------------------------------
    options = apriltag.DetectorOptions(refine_pose=True)
    detector = apriltag.Detector(options)

    # -------------------------------------------------------
    # 4. Load image
    # -------------------------------------------------------
    img_path = "debug.png"
    print("[INFO] Reading:", img_path)

    frame = cv2.imread(img_path)
    if frame is None:
        print("❌ Cannot read image!")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------
    # 5. Detect on fisheye image (NO undistort)
    # -------------------------------------------------------
    detections, dbg = detector.detect(gray, return_image=True)
    print(f"[INFO] Found {len(detections)} tags")

    overlay = frame.copy()
    h, w = frame.shape[:2]

    for det in detections:
        tag_id = det.tag_id
        center = det.center.astype(int)
        corners = det.corners  # (4,2)

        print(f" → Tag ID = {tag_id}, center = {center}")

        # Estimate pose using fisheye model
        T, rvec, tvec = apritag_pose_fisheye(corners, tag_size, K, D)
        T = convert_pose_new_to_old(T)

        if T is None:
            print("❌ PnP failed!")
            continue

        print("[POSE]\n", T)

        cv2.putText(
            overlay, f"ID: {tag_id}",
            (center[0], center[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2
        )

        for pt in corners.astype(int):
            cv2.circle(overlay, tuple(pt), 4, (0, 255, 0), -1)

    # -------------------------------------------------------
    # 6. Display
    # -------------------------------------------------------
    cv2.imshow("Frame", frame)
    cv2.imshow("Detections", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
