"""
🏗️ Smart Tower Builder - Main
==============================
비전 + 로봇 통합 실행 파일

사용법:
    python main.py
    
조작:
    - 블록 클릭: 해당 블록 집어서 옮기기
    - 빈 공간 클릭: 해당 위치 깊이 확인
    - 'h': 홈 위치로 이동
    - 'o': 그리퍼 열기
    - 'c': 그리퍼 닫기
    - ESC: 종료
"""

import cv2
import rclpy
import numpy as np
from typing import Optional

from camera import BlockDetectionSystem, Block, DetectorConfig
from controller import RobotController, RobotConfig

import DR_init


# ============================================================
# 설정
# ============================================================

ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"


# ============================================================
# 메인 애플리케이션
# ============================================================

class SmartTowerApp:
    """
    Smart Tower Builder 메인 애플리케이션
    
    비전 시스템과 로봇 컨트롤러를 통합
    """
    
    def __init__(self):
        self.vision: Optional[BlockDetectionSystem] = None
        self.robot: Optional[RobotController] = None
        self._dsr_node = None
        self._selected_block: Optional[Block] = None
    
    def initialize(self) -> bool:
        """시스템 초기화"""
        print("\n" + "=" * 50)
        print("🏗️ Smart Tower Builder")
        print("=" * 50)
        
        # 1. ROS2 초기화
        print("\n[1/3] ROS2 초기화...")
        rclpy.init()
        self._dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
        DR_init.__dsr__id = ROBOT_ID
        DR_init.__dsr__model = ROBOT_MODEL
        DR_init.__dsr__node = self._dsr_node
        
        # 2. 비전 시스템 초기화
        print("[2/3] 비전 시스템 초기화...")
        self.vision = BlockDetectionSystem()
        if not self.vision.start():
            print("❌ 비전 시스템 초기화 실패")
            return False
        
        # 3. 로봇 초기화
        print("[3/3] 로봇 초기화...")
        robot_config = RobotConfig(
            robot_id=ROBOT_ID,
            robot_model=ROBOT_MODEL,
        )
        self.robot = RobotController(robot_config)
        if not self.robot.initialize():
            print("❌ 로봇 초기화 실패")
            return False
        
        print("\n✅ 모든 시스템 초기화 완료!")
        print("=" * 50)
        return True
    
    def shutdown(self):
        """시스템 종료"""
        print("\n프로그램을 종료합니다...")
        
        if self.robot:
            self.robot.shutdown()
            self.robot.destroy_node()
        
        if self.vision:
            self.vision.stop()
        
        if self._dsr_node:
            self._dsr_node.destroy_node()
        
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("종료 완료.")
    
    # -------------------- GUI --------------------
    
    def setup_gui(self):
        """GUI 윈도우 및 트랙바 설정"""
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", 640, 480)
        cv2.setMouseCallback("Result", self._on_mouse)
        
        cv2.namedWindow("Control")
        cfg = self.vision.config
        cv2.createTrackbar("Threshold", "Control", cfg.threshold, 255, lambda x: None)
        cv2.createTrackbar("Min Area", "Control", cfg.min_area, 5000, lambda x: None)
        cv2.createTrackbar("Max Area", "Control", cfg.max_area, 30000, lambda x: None)
        
        print("\n📌 조작법:")
        print("   - 블록 클릭: 집어서 옮기기")
        print("   - 빈 공간 클릭: 깊이 확인")
        print("   - 'h': 홈 위치")
        print("   - 'o'/'c': 그리퍼 열기/닫기")
        print("   - ESC: 종료")
        print("=" * 50 + "\n")
    
    def _on_mouse(self, event, x, y, flags, param):
        """마우스 클릭 콜백"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        # 블록 찾기 (rotated_box 기반으로 정확한 영역 체크)
        block = self.vision.find_block_at(x, y, update=False)
        
        if block:
            # 블록 클릭 → 집어서 옮기기
            self._selected_block = block
            print(f"\n🎯 블록 선택: {block}")
            
            if block.is_valid:
                coords = self.robot.block_to_robot_coords(block)
                if coords:
                    print(f"   로봇 좌표: X={coords[0]:.1f}, Y={coords[1]:.1f}, Z={coords[2]:.1f}")
                    
                    # 블록 집어서 옮기기
                    self.robot.pick_and_place(
                        block,
                        place_offset=(0, 100, 0),  # Y 방향으로 100mm 이동
                        return_home=True
                    )
            else:
                print("   ⚠️ 깊이 정보 없음")
        else:
            # 빈 공간 클릭 → 깊이 확인
            self._selected_block = None
            depth = self.vision.get_depth_at(x, y)
            print(f"\n빈 공간 클릭 ({x}, {y})")
            if depth > 0:
                print(f"   깊이: {depth * 100:.1f} cm")
                point_3d = self.vision.get_3d_at(x, y)
                if point_3d:
                    print(f"   3D: X={point_3d[0]*1000:.1f}, Y={point_3d[1]*1000:.1f}, Z={point_3d[2]*1000:.1f} mm")
            else:
                print("   깊이 측정 불가")
    
    def _handle_key(self, key: int) -> bool:
        """
        키 입력 처리
        
        Returns:
            False면 종료
        """
        if key == 27:  # ESC
            return False
        elif key == ord('h'):
            print("🏠 홈 위치로 이동...")
            self.robot.move_home()
        elif key == ord('o'):
            print("🖐️ 그리퍼 열기")
            self.robot.gripper_open()
        elif key == ord('c'):
            print("✊ 그리퍼 닫기")
            self.robot.gripper_close()
        
        return True
    
    # -------------------- 메인 루프 --------------------
    
    def run(self):
        """메인 루프 실행"""
        self.setup_gui()
        
        try:
            while rclpy.ok():
                # ROS2 스핀
                rclpy.spin_once(self.robot, timeout_sec=0.001)
                rclpy.spin_once(self._dsr_node, timeout_sec=0.001)
                
                # 트랙바 값 반영
                cfg = self.vision.config
                cfg.threshold = cv2.getTrackbarPos("Threshold", "Control")
                cfg.min_area = cv2.getTrackbarPos("Min Area", "Control")
                cfg.max_area = cv2.getTrackbarPos("Max Area", "Control")
                
                # 비전 업데이트
                if not self.vision.update():
                    continue
                
                # 화면 그리기
                display = self._draw_frame()
                cv2.imshow("Result", display)
                
                # 이진화 이미지
                if self.vision.detector.binary_view is not None:
                    cv2.imshow("Binary (ROI)", self.vision.detector.binary_view)
                
                # 깊이 이미지
                depth_display = self._draw_depth()
                cv2.imshow("Depth", depth_display)
                
                # 키 입력
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break
                    
        except KeyboardInterrupt:
            print("\nCtrl+C로 종료...")
    
    def _draw_frame(self) -> np.ndarray:
        """결과 프레임 그리기"""
        frame = self.vision.last_frame.copy()
        cfg = self.vision.config
        blocks = self.vision.last_blocks
        
        # ROI
        cv2.rectangle(frame, 
                     (cfg.roi_x, cfg.roi_y),
                     (cfg.roi_x + cfg.roi_w, cfg.roi_y + cfg.roi_h),
                     (0, 0, 255), 2)
        cv2.putText(frame, "ROI", (cfg.roi_x, cfg.roi_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 블록들
        for block in blocks:
            is_selected = (block == self._selected_block)
            color = (0, 255, 255) if is_selected else (0, 255, 0)
            thickness = 3 if is_selected else 2
            
            cv2.drawContours(frame, [block.rotated_box], 0, color, thickness)
            
            cx, cy = block.center_2d
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            if block.is_valid:
                cv2.putText(frame, f"{block.depth_cm:.0f}cm", (cx - 15, cy - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"{block.real_width_mm:.0f}x{block.real_height_mm:.0f}mm", 
                           (cx - 30, cy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "no depth", (cx - 25, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        cv2.putText(frame, f"Blocks: {len(blocks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _draw_depth(self) -> np.ndarray:
        """깊이 이미지 시각화"""
        depth_img = self.vision.last_depth
        
        depth_display = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        valid_mask = depth_img > 0
        if np.any(valid_mask):
            scale = self.vision.camera.depth_scale
            min_d = np.min(depth_img[valid_mask]) * scale * 100
            max_d = np.max(depth_img[valid_mask]) * scale * 100
            cv2.putText(depth_display, f"Range: {min_d:.0f}-{max_d:.0f}cm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return depth_display


# ============================================================
# 엔트리 포인트
# ============================================================

def main():
    app = SmartTowerApp()
    
    try:
        if app.initialize():
            app.run()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()