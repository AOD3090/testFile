import cv2
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np

import DR_init
from tower.gripper_drl_controller import GripperController

# ✅ VisionBlockDetector에서 RealSense + Detector 가져오기
from tower.VisionBlockDetector import RealSenseCamera, BlockDetector

ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 200, 100

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")
        self.get_logger().info("RobotControllerNode 시작")

        # =========================
        # ✅ RealSense는 VisionBlockDetector 방식(파이썬 pyrealsense2)으로 직접 구동
        # =========================
        self.rs_cam = RealSenseCamera(width=640, height=480, fps=30)
        if not self.rs_cam.start():
            self.get_logger().error("RealSenseCamera start() 실패")
            raise RuntimeError("RealSense start failed")

        self.intrinsics = self.rs_cam.intrinsics
        self.depth_scale = self.rs_cam.depth_scale

        self.latest_cv_color = None
        self.latest_cv_depth_mm = None

        # =========================
        # ✅ 블록 디텍터(ROI/Threshold/MinArea/MaxArea + binary_view)
        # =========================
        self.detector = BlockDetector()
        self.blocks = []
        self.selected_idx = -1

        # =========================
        # 로봇 및 그리퍼 초기화 (기존 유지)
        # =========================
        self.gripper = None
        try:
            from DSR_ROBOT2 import wait
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            self.get_logger().info("Waiting for service /dsr01/drl/drl_start...")
            wait(2)

            if not self.gripper.initialize():
                self.get_logger().error("Gripper initialization failed. Exiting.")
                raise Exception("Gripper initialization failed")

            self.get_logger().info("그리퍼를 활성화합니다...")
            self.gripper_is_open = True
            self.gripper.move(0)
        except Exception as e:
            self.get_logger().error(f"An error occurred during gripper setup: {e}")
            pass

        self.get_logger().info("초기화 완료")

    def stop_camera(self):
        if self.rs_cam:
            self.rs_cam.stop()

    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()

    # =========================
    # ✅ 매 프레임 처리: 프레임 수신 → detect → 창 출력
    # =========================
    def process_and_render(self):
        # 트랙바 값 반영
        self.detector.threshold = cv2.getTrackbarPos("Threshold", "Control")
        self.detector.min_area = cv2.getTrackbarPos("Min Area", "Control")
        self.detector.max_area = cv2.getTrackbarPos("Max Area", "Control")

        color, depth_img, _depth_frame = self.rs_cam.get_frames()
        if color is None or depth_img is None:
            return

        self.latest_cv_color = color
        self.latest_cv_depth_mm = depth_img  # uint16 raw depth

        # detect
        self.blocks = self.detector.detect(color, depth_img, self.rs_cam)

        # ===== Result 창 (ROI + 박스 + 수치) =====
        display = color.copy()
        d = self.detector

        # ROI 박스
        cv2.rectangle(display, (d.roi_x, d.roi_y), (d.roi_x + d.roi_w, d.roi_y + d.roi_h), (0, 0, 255), 2)
        cv2.putText(display, "ROI (WHITE)", (d.roi_x, d.roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 블록 표시
        for i, block in enumerate(self.blocks):
            is_sel = (i == self.selected_idx)
            color_draw = (0, 255, 255) if is_sel else (0, 255, 0)
            thickness = 3 if is_sel else 2

            cv2.drawContours(display, [block.rotated_box], 0, color_draw, thickness)

            cx, cy = block.center_2d
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

            if block.depth > 0:
                txt_depth = f"{block.depth * 100:.0f}cm"
                cv2.putText(display, txt_depth, (cx - 15, cy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_draw, 2)

                txt_size = f"{block.real_width_mm:.0f}x{block.real_height_mm:.0f}mm"
                cv2.putText(display, txt_size, (cx - 30, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                cv2.putText(display, "no depth", (cx - 25, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.putText(display, f"({cx},{cy})", (cx - 25, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(display, f"Blocks: {len(self.blocks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Result", display)

        # ===== Binary (ROI) 창 =====
        if self.detector.binary_view is not None:
            cv2.imshow("Binary (ROI)", self.detector.binary_view)

        # ===== Depth 창 =====
        depth_display = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03),
            cv2.COLORMAP_JET
        )

        valid_mask = depth_img > 0
        if np.any(valid_mask):
            min_d = np.min(depth_img[valid_mask]) * self.depth_scale * 100  # cm
            max_d = np.max(depth_img[valid_mask]) * self.depth_scale * 100  # cm
            cv2.putText(depth_display, f"Range: {min_d:.0f}-{max_d:.0f}cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Depth", depth_display)

    # =========================
    # ✅ Result 창 클릭: (1) 블록 선택(하이라이트) + (2) 로봇 이동
    # - 블록 클릭이면: 블록 중심으로 이동
    # - 빈 공간 클릭이면: 클릭 지점으로 이동
    # =========================
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.latest_cv_depth_mm is None or self.intrinsics is None:
            self.get_logger().warn("아직 뎁스 프레임 또는 intrinsics가 준비되지 않았습니다.")
            return

        # 블록 선택
        self.selected_idx = -1
        target_u, target_v = x, y

        for i, b in enumerate(self.blocks):
            bx, by, bw, bh = b.bbox
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.selected_idx = i
                target_u, target_v = b.center_2d  # 블록 중심으로 이동
                break

        # depth 읽기 (배열 방식)
        depth_m = self.rs_cam.get_depth_from_image(target_u, target_v, self.latest_cv_depth_mm, debug=False)
        if depth_m <= 0:
            print(f"({target_u}, {target_v}) 지점의 깊이를 측정할 수 없습니다.")
            return

        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [target_u, target_v], depth_m)

        # RealSense (x, y, z)[m] -> mm
        rs_x_mm = point_3d[0] * 1000  # 오른쪽
        rs_y_mm = point_3d[1] * 1000  # 아래
        rs_z_mm = point_3d[2] * 1000  # 앞(거리)

        # ---- 사용자님 기존 변환 로직 유지 ----
        final_x = 685 + rs_y_mm
        final_y = rs_x_mm + 20
        final_z = 810 - rs_z_mm

        if final_z <= 150:
            final_z = 150

        print("--- 변환된 최종 3D 좌표 ---")
        print(f"픽셀 좌표: (u={target_u}, v={target_v}), Depth: {depth_m*1000:.1f} mm")
        print(f"로봇 목표 좌표: X={final_x:.1f}, Y={final_y:.1f}, Z={final_z:.1f}\n")

        self.move_robot_and_control_gripper(final_x, final_y, final_z)
        print("=" * 50)

    def move_robot_and_control_gripper(self, x, y, z):
        from DSR_ROBOT2 import get_current_posx, movel, wait, movej
        from DR_common2 import posx, posj

        try:
            current_pos = get_current_posx()[0]
            cur_x, cur_y, cur_z, cur_Rx, cur_Ry, cur_Rz = current_pos

            approach_lift_mm = 50.0

            target_up = [x, y, z + approach_lift_mm, cur_Rx, cur_Ry, cur_Rz]
            target_at = [x, y, z, cur_Rx, cur_Ry, cur_Rz]
            home_pose = posj(0, 0, 90, 0, 90, 0)

            movel(posx(target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            self.get_logger().info(f"→ 회전 없이 접근/하강 이동: {target_at}")
            movel(posx(target_at), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 그리퍼 닫기 (집기)
            if self.gripper:
                self.gripper.move(550)
            wait(1.5)

            # 다시 들어올리기
            movel(posx(target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 드롭 지점
            drop_offset = 100.0
            drop_target_up = [x, y + drop_offset, z + approach_lift_mm, cur_Rx, cur_Ry, cur_Rz]
            movel(posx(drop_target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            drop_target_down = [x, y + drop_offset, z, cur_Rx, cur_Ry, cur_Rz]
            movel(posx(drop_target_down), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 그리퍼 열기 (놓기)
            if self.gripper:
                self.gripper.move(100)
            wait(1.0)

            movel(posx(drop_target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            self.get_logger().info(":집: 홈 자세로 복귀합니다.")
            movej(home_pose, VELOCITY, ACC)
            wait(1.0)

        except Exception as e:
            self.get_logger().error(f"로봇 이동 및 그리퍼 제어 중 오류 발생: {e}")


def main(args=None):
    rclpy.init(args=args)

    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    try:
        from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    except ImportError as e:
        print(f"DSR_ROBOT2 라이브러리를 임포트할 수 없습니다: {e}")
        rclpy.shutdown()
        return

    robot_controller = RobotControllerNode()

    # ===== 창/트랙바 구성 (VisionBlockDetector 스타일) =====
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 640, 480)
    cv2.setMouseCallback("Result", robot_controller.mouse_callback)

    cv2.namedWindow("Control")
    cv2.createTrackbar("Threshold", "Control", robot_controller.detector.threshold, 255, lambda x: None)
    cv2.createTrackbar("Min Area", "Control", robot_controller.detector.min_area, 5000, lambda x: None)
    cv2.createTrackbar("Max Area", "Control", robot_controller.detector.max_area, 30000, lambda x: None)

    print("Result 창에서 클릭하세요. (블록 클릭: 블록 중심 / 빈 공간 클릭: 해당 픽셀)  ESC 종료")

    try:
        while rclpy.ok():
            rclpy.spin_once(robot_controller, timeout_sec=0.001)
            rclpy.spin_once(dsr_node, timeout_sec=0.001)

            robot_controller.process_and_render()

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Ctrl+C로 종료합니다...")
    finally:
        print("프로그램을 종료합니다...")
        robot_controller.terminate_gripper()
        robot_controller.stop_camera()
        cv2.destroyAllWindows()
        robot_controller.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()
        print("종료 완료.")


if __name__ == "__main__":
    main()
