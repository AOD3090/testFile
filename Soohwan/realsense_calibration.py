import pyrealsense2 as rs
import cv2
import numpy as np

# =====================
# 체커보드 설정
# =====================
chessboard_size = (7, 4)
square_size = 0.025  # meters

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                       0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

# =====================
# RealSense 초기화
# =====================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

print("체커보드를 다양한 각도로 보여주고 'c'를 눌러 캡처하세요")
print("충분히 모였으면 'q'로 종료")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray,
            chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = img.copy()

        if ret:
            cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS +
                          cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            cv2.drawChessboardCorners(display, chessboard_size, corners, ret)

        cv2.imshow("RealSense Chessboard", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"Captured frame {len(objpoints)}")

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# =====================
# Camera Calibration
# =====================
if len(objpoints) < 5:
    print("캘리브레이션용 이미지가 부족합니다.")
    exit()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

# =====================
# Reprojection Error
# =====================
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist
    )
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Reprojection Error: {mean_error / len(objpoints)}")
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)

np.savez(
    "realsense_camera_calibration.npz",
    mtx=mtx,
    dist=dist,
    rvecs=rvecs,
    tvecs=tvecs
)

print("Calibration data saved: realsense_camera_calibration.npz")
