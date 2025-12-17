import cv2
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import time
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import DR_init
from tower.gripper_drl_controller import GripperController

ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 200, 200

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotControllerNode(Node):
    
    def __init__(self):
        super().__init__("robot_controller_node")
        self.bridge = CvBridge()
        self.get_logger().info("ROS 2 구독자 설정을 시작합니다...")
        self.intrinsics = None
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None
        # :압정: 수정: 토픽 리스트 기반, 네임스페이스('/camera/camera/')가 붙은 토픽으로 통일
        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw'  # 토픽 리스트에 존재하는 네임스페이스
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw' # 토픽 리스트에 존재하는 네임스페이스
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info' # 토픽 리스트에 존재하는 네임스페이스
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)
        self.get_logger().info("컬러/뎁스/카메라정보 토픽 구독 대기 중...")
        self.get_logger().info("화면이 나오지 않으면 RealSense Launch 명령어를 확인하세요.")
        # --- 로봇 및 그리퍼 초기화 (생략 없이 동일) ---
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
        self.get_logger().info("RealSense ROS 2 구독자와 로봇 컨트롤러가 초기화되었습니다.")
        # ---------------------------

    def synced_callback(self, color_msg, depth_msg, info_msg):
        try:
            self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge 변환 오류: {e}")
            return
        # Intrinsics 설정은 변화 없음
        if self.intrinsics is None:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width = info_msg.width
            self.intrinsics.height = info_msg.height
            self.intrinsics.ppx = info_msg.k[2]
            self.intrinsics.ppy = info_msg.k[5]
            self.intrinsics.fx = info_msg.k[0]
            self.intrinsics.fy = info_msg.k[4]
            if info_msg.distortion_model == 'plumb_bob' or info_msg.distortion_model == 'rational_polynomial':
                self.intrinsics.model = rs.distortion.brown_conrady
            else:
                self.intrinsics.model = rs.distortion.none
            self.intrinsics.coeffs = list(info_msg.d)
            self.get_logger().info("카메라 내장 파라미터(Intrinsics) 수신 완료.")
    # stop_camera, terminate_gripper는 동일

    def stop_camera(self):
        pass

    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()

    def mouse_callback(self, event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.latest_cv_depth_mm is None or self.intrinsics is None:
                self.get_logger().warn("아직 뎁스 프레임 또는 카메라 정보가 수신되지 않았습니다.")
                return
            try:
                depth_mm = self.latest_cv_depth_mm[v, u]
            except IndexError:
                self.get_logger().warn(f"클릭 좌표(u={u}, v={v})가 이미지 범위를 벗어났습니다.")
                return
            if depth_mm == 0:
                print(f"({u}, {v}) 지점의 깊이를 측정할 수 없습니다 (값: 0).")
                return
            depth_m = float(depth_mm) / 1000.0
            point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth_m)
            # :압정: 수정: 로봇 좌표계 변환 로직 보강
            # RealSense (x, y, z) -> 로봇 좌표계 (X, Y, Z)
            # RealSense: X(오른쪽), Y(아래), Z(앞)
            # DSR 로봇: X(앞), Y(왼쪽), Z(위)
            # 1. RealSense의 3D 좌표 (mm)
            rs_x_mm = point_3d[0] * 1000  # (오른쪽)
            rs_y_mm = point_3d[1] * 1000  # (아래)
            rs_z_mm = point_3d[2] * 1000  # (앞/거리)
            # 2. 로봇 좌표계로 변환 (카메라 위치: X=1000, Z=1200에서 아래를 보고 있다고 가정)
            # 이 변환 공식은 캘리브레이션에 따라 다르므로, 현재 코드의 공식을 유지하되,
            # 카메라 위치 정보를 바탕으로 일반적인 변환 로직을 주석으로 명시합니다.
            # --- 강사님의 기존 로직 유지 (캘리브레이션 값으로 추정) ---
            final_x = 685 + rs_y_mm # RealSense Y(아래)가 로봇 X(앞)에 대응? (Y축 회전)
            final_y = rs_x_mm + 20            # RealSense X(오른쪽)가 로봇 Y(왼쪽)에 대응? (Y축 회전)
            final_z = 810 - rs_z_mm # RealSense Z(깊이)가 로봇 Z(높이)에 대응?
            # x - 660    y - 34     z - 821
            # --------------------------------------------------------
            if(final_z <= 150):
                final_z = 150
            print("--- 변환된 최종 3D 좌표 ---")
            print(f"픽셀 좌표: (u={u}, v={v}), Depth: {depth_m*1000:.1f} mm")
            print(f"로봇 목표 좌표: X={final_x:.1f}, Y={final_y:.1f}, Z={final_z:.1f}\n")
            self.move_robot_and_control_gripper(final_x, final_y, final_z)
            print("=" * 50)
    # move_robot_and_control_gripper 함수 및 main 함수는 동일

    def move_robot_and_control_gripper(self, x, y, z):
        from DSR_ROBOT2 import get_current_posx, movel, wait, movej
        from DR_common2 import posx, posj
        try:
            current_pos = get_current_posx()[0]
            cur_x, cur_y, cur_z, cur_Rx, cur_Ry, cur_Rz = current_pos
            
            approach_lift_mm = 50.0
            
            target_up = [x, y, z + approach_lift_mm, cur_Rx, cur_Ry, cur_Rz]   # :흰색_확인_표시: [MOD]
            target_at = [x, y, z,                    cur_Rx, cur_Ry, cur_Rz]   # :흰색_확인_표시: [MOD]
            home_pose = posj(0, 0, 90, 0, 90, 0)
            # 단계별 이동
            movel(posx(target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)
            
            self.get_logger().info(f"→ 회전 없이 접근/하강 이동: {target_at}")  # :흰색_확인_표시: [MOD]
            movel(posx(target_at), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 그리퍼 닫기 (집기)
            self.gripper.move(550)
            wait(1.5)

            # 다시 들어올리기
            movel(posx(target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)
            
            # 드롭 지점(조금 옆으로 이동) - 기존 로직 유지
            drop_offset = 100.0

            # approach_axis 없이 dx/dy 비교도 제거 (회전 로직과 함께 필요 없음)
            # 간단히 y 방향으로만 이동하고 싶으면 아래처럼 고정할 수도 있습니다.
            drop_target_up = [x, y + drop_offset, z + approach_lift_mm, cur_Rx, cur_Ry, cur_Rz]  # :흰색_확인_표시: [MOD]
            movel(posx(drop_target_up), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # "처음 내려간 만큼" 다시 내려가서(= approach_lift_mm 만큼) 그리퍼 열기
            drop_target_down = [x, y + drop_offset, z, cur_Rx, cur_Ry, cur_Rz]  # :흰색_확인_표시: [MOD]
            movel(posx(drop_target_down), vel=VELOCITY, acc=ACC)                # :흰색_확인_표시: [MOD]
            wait(0.3)

            # 그리퍼 열기 (놓기)
            self.gripper.move(100)
            wait(1.0)
            # 다시 위로 살짝 올리고 복귀(선택이지만 충돌 방지에 유리)
            movel(posx(drop_target_up), vel=VELOCITY, acc=ACC)  # :흰색_확인_표시: [MOD]
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
        from DSR_ROBOT2 import get_current_posx, movel, wait, movej
        from DR_common2 import posx, posj
    except ImportError as e:
        print(f"DSR_ROBOT2 라이브러리를 임포트할 수 없습니다: {e}")
        rclpy.shutdown()
        exit(1)
    robot_controller = RobotControllerNode()
    # :압정: 창 크기 조정 코드 유지
    cv2.namedWindow("RealSense Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RealSense Camera", 640, 480)
    cv2.setMouseCallback("RealSense Camera", robot_controller.mouse_callback)
    print("카메라 영상에서 원하는 지점을 클릭하세요. 'ESC' 키를 누르면 종료됩니다.")

    try:
        while rclpy.ok():
            rclpy.spin_once(robot_controller, timeout_sec=0.001)
            rclpy.spin_once(dsr_node, timeout_sec=0.001)
            if robot_controller.latest_cv_color is not None:
                display_image = robot_controller.latest_cv_color.copy()
                h, w, _ = display_image.shape
                # 중앙점 표시
                cv2.circle(display_image, (w // 2, h // 2), 3, (0, 0, 255), -1)
                cv2.imshow("RealSense Camera", display_image)
            # 이미지 수신 여부와 관계없이 cv2.waitKey는 호출되어야 GUI 이벤트 처리가 됩니다.
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        print("Ctrl+C로 종료합니다...")
    except Exception as e:
        print(f"실행 중 예외 발생: {e}")
    finally:
        print("프로그램을 종료합니다...")
        robot_controller.terminate_gripper()
        cv2.destroyAllWindows()
        robot_controller.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()
        print("종료 완료.")

if __name__ == '__main__':
    main()