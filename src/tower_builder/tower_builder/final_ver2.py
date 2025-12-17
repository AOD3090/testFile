"""
🏗️ Smart Tower Builder - 수정 버전 v2
===========================================
수정 사항:
1. Binary 디버그 창 추가
2. depth 필터링 제대로 적용
3. ROI 외부 검출 방지
4. 디버깅 출력 개선
"""
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
from tower_builder.gripper_drl_controller import GripperController
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import deque
from enum import Enum

# 로봇 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 50, 50
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotState(Enum):
    IDLE = 0
    BLOCK_SELECTED = 1
    PICKING = 2
    PLACING = 3


@dataclass
class Block:
    bbox: Tuple[int, int, int, int]
    center_2d: Tuple[int, int]
    contour: np.ndarray = field(compare=False)
    rotated_box: np.ndarray = field(compare=False)
    area: float = 0.0
    aspect_ratio: float = 0.0
    solidity: float = 0.0
    center_3d: Optional[Tuple[float, float, float]] = None
    depth: float = 0.0
    real_width_mm: float = 0.0
    real_height_mm: float = 0.0
    size_class: str = "unknown"
    stability_score: int = 0
    last_seen_time: float = 0.0
    missing_frames: int = 0
    is_interpolated: bool = False


class BlockDetector:
    def __init__(self):
        # 적응형 이진화 설정
        self.use_adaptive = True
        self.adaptive_block_size = 15
        self.adaptive_c = 1
        self.use_inv = True
        
        self.multi_blur = True
        self.blur_sizes = [(3, 3), (5, 5)]
        
        # 크기 필터
        self.min_area = 50
        self.max_area = 15000
        
        # ROI 설정
        self.roi_x = 150
        self.roi_y = 120
        self.roi_w = 350
        self.roi_h = 250
        
        # 형태 필터
        self.min_aspect = 0.2
        self.max_aspect = 6.0
        self.min_solidity = 0.3
        
        # 🔥 depth 필터 (핵심!)
        self.min_depth = 0.3   # 300mm 이상
        self.max_depth = 1.2   # 1200mm 이하 (배경 제거)
        
        # 블록 크기 분류 기준 (mm)
        self.small_thresh = 30
        self.medium_thresh = 50
        
        # 트래킹 설정
        self.tracked_blocks = []
        self.max_track_distance = 50
        self.stability_threshold = 2
        self.max_missing_frames = 5
        
        # 디버그용
        self.binary_view = None
        self.debug_frame = None
        
    def detect(self, frame, depth_image, intrinsics, depth_scale) -> List[Block]:
        current_time = time.time()
        all_candidates = []
        
        # 적응형 이진화로 검출
        contours, binary = self._detect_with_adaptive(frame)
        self.binary_view = binary
        
        # 디버그 프레임 생성
        self.debug_frame = frame.copy()
        
        print(f"\n🔍 검출된 윤곽선 수: {len(contours)}")
        
        for cnt in contours:
            candidate = self._process_contour(cnt, current_time)
            if candidate:
                all_candidates.append(candidate)
        
        print(f"✅ 형태 필터 통과: {len(all_candidates)}")
        
        unique_candidates = self._merge_duplicates(all_candidates)
        print(f"🔄 중복 제거 후: {len(unique_candidates)}")
        
        blocks = []
        depth_rejected = 0
        
        for cand in unique_candidates:
            cx, cy = cand['center']
            depth = self._get_depth_from_image(cx, cy, depth_image, depth_scale)
            
            # 🔥 depth 필터링 먼저!
            if depth <= 0:
                print(f"  ❌ ({cx}, {cy}): depth=0 (무효)")
                depth_rejected += 1
                continue
                
            if not (self.min_depth < depth < self.max_depth):
                print(f"  ❌ ({cx}, {cy}): depth={depth*1000:.0f}mm (범위 밖: {self.min_depth*1000:.0f}~{self.max_depth*1000:.0f}mm)")
                depth_rejected += 1
                continue
            
            # depth 유효한 경우만 블록 생성
            block = Block(
                bbox=cand['bbox'],
                center_2d=cand['center'],
                contour=cand['contour'],
                rotated_box=cand['rotated_box'],
                area=cand['area'],
                aspect_ratio=cand['aspect_ratio'],
                solidity=cand['solidity'],
                last_seen_time=current_time
            )
            
            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
            block.center_3d = point_3d
            block.depth = depth
            
            real_w, real_h = self._calc_real_size(
                cand['rect_w'], cand['rect_h'], depth, intrinsics
            )
            block.real_width_mm = real_w
            block.real_height_mm = real_h
            
            avg = (real_w + real_h) / 2
            if avg < self.small_thresh:
                block.size_class = "small"
            elif avg < self.medium_thresh:
                block.size_class = "medium"
            else:
                block.size_class = "large"
            
            print(f"  ✅ ({cx}, {cy}): depth={depth*1000:.0f}mm, size={block.size_class}")
            blocks.append(block)
        
        print(f"⛔ depth 필터로 제거: {depth_rejected}")
        
        blocks = self._track_blocks(blocks, current_time)
        
        stable_count = len([b for b in blocks if b.stability_score >= self.stability_threshold])
        print(f"🎯 최종 블록 수: {len(blocks)} (안정: {stable_count})")
        
        return blocks
    
    def _detect_with_adaptive(self, frame):
        roi = frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        all_contours = []
        final_binary = None
        
        for blur_size in self.blur_sizes:
            blurred = cv2.GaussianBlur(gray, blur_size, 0)
            
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY if self.use_inv else cv2.THRESH_BINARY_INV,
                self.adaptive_block_size,
                self.adaptive_c
            )
            
            if blur_size == (5, 5):
                kernel = np.ones((2, 2), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                final_binary = binary.copy()
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        if final_binary is None:
            final_binary = binary
        
        return all_contours, final_binary
    
    def _process_contour(self, cnt, current_time):
        area = cv2.contourArea(cnt)
        is_small = area < 2000
        
        if is_small:
            if area < 30:
                return None
        else:
            if not (self.min_area < area < self.max_area):
                return None
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (_, _), (w, h), _ = rect
        
        if h == 0 or w == 0:
            return None
        
        aspect = max(w, h) / min(w, h)
        
        if is_small:
            if not (0.1 <= aspect <= 10.0):
                return None
        else:
            if not (self.min_aspect <= aspect <= self.max_aspect):
                return None
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return None
        
        solidity = area / hull_area
        min_sol = 0.2 if is_small else self.min_solidity
        if solidity < min_sol:
            return None
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if not (3 <= len(approx) <= 12):
            return None
        
        # ROI 오프셋 적용
        box_global = box.copy()
        box_global[:, 0] += self.roi_x
        box_global[:, 1] += self.roi_y
        cnt_global = cnt.copy()
        cnt_global[:, :, 0] += self.roi_x
        cnt_global[:, :, 1] += self.roi_y
        
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"]) + self.roi_x
        cy = int(M["m01"] / M["m00"]) + self.roi_y
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        return {
            'bbox': (x + self.roi_x, y + self.roi_y, bw, bh),
            'center': (cx, cy),
            'contour': cnt_global,
            'rotated_box': box_global,
            'area': area,
            'aspect_ratio': aspect,
            'solidity': solidity,
            'rect_w': w,
            'rect_h': h,
            'is_small': is_small
        }
    
    def _merge_duplicates(self, candidates):
        if not candidates:
            return []
        
        unique = []
        used = [False] * len(candidates)
        
        for i, cand1 in enumerate(candidates):
            if used[i]:
                continue
            
            cx1, cy1 = cand1['center']
            group = [cand1]
            used[i] = True
            
            for j, cand2 in enumerate(candidates):
                if used[j]:
                    continue
                
                cx2, cy2 = cand2['center']
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                if dist < 30:
                    group.append(cand2)
                    used[j] = True
            
            best = max(group, key=lambda x: x['area'])
            unique.append(best)
        
        return unique
    
    def _track_blocks(self, current_blocks, current_time):
        if not self.tracked_blocks:
            for block in current_blocks:
                block.stability_score = 1
                block.missing_frames = 0
            self.tracked_blocks = current_blocks
            return current_blocks
        
        matched = [False] * len(current_blocks)
        updated_tracked = []
        
        for old_block in self.tracked_blocks:
            old_cx, old_cy = old_block.center_2d
            best_idx = -1
            best_dist = float('inf')
            
            for i, curr_block in enumerate(current_blocks):
                if matched[i]:
                    continue
                
                curr_cx, curr_cy = curr_block.center_2d
                dist = np.sqrt((old_cx - curr_cx)**2 + (old_cy - curr_cy)**2)
                
                if dist < self.max_track_distance and dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            if best_idx >= 0:
                matched[best_idx] = True
                current_blocks[best_idx].stability_score = min(old_block.stability_score + 1, 30)
                current_blocks[best_idx].missing_frames = 0
                current_blocks[best_idx].is_interpolated = False
                updated_tracked.append(current_blocks[best_idx])
            else:
                old_block.missing_frames += 1
                
                if old_block.missing_frames <= self.max_missing_frames:
                    old_block.is_interpolated = True
                    old_block.last_seen_time = current_time
                    old_block.stability_score = max(old_block.stability_score - 1, 0)
                    updated_tracked.append(old_block)
        
        for i, block in enumerate(current_blocks):
            if not matched[i]:
                block.stability_score = 1
                block.missing_frames = 0
                block.is_interpolated = False
                updated_tracked.append(block)
        
        self.tracked_blocks = updated_tracked
        return updated_tracked
    
    def _get_depth_from_image(self, x, y, depth_image, depth_scale):
        h, w = depth_image.shape
        x, y = int(x), int(y)
        
        if not (0 <= x < w and 0 <= y < h):
            return 0.0
        
        raw_depth = depth_image[y, x]
        depth_m = raw_depth * depth_scale
        
        if depth_m > 0.05:
            return depth_m
        
        sample_offsets = [
            (-5, 0), (5, 0), (0, -5), (0, 5),
            (-10, 0), (10, 0), (0, -10), (0, 10),
            (-15, 0), (15, 0), (0, -15), (0, 15),
        ]
        
        valid_depths = []
        for dx, dy in sample_offsets:
            sx, sy = x + dx, y + dy
            if 0 <= sx < w and 0 <= sy < h:
                raw = depth_image[sy, sx]
                d = raw * depth_scale
                if 0.05 < d < 3.0:
                    valid_depths.append(d)
        
        if valid_depths:
            valid_depths.sort()
            return valid_depths[len(valid_depths) // 2]
        
        return 0.0
    
    def _calc_real_size(self, w_px, h_px, depth, intrinsics):
        if depth <= 0:
            return 0, 0
        real_w = (w_px * depth * 1000) / intrinsics.fx
        real_h = (h_px * depth * 1000) / intrinsics.fy
        return real_w, real_h
    
    def find_block_at(self, blocks, x, y):
        for b in blocks:
            if b.is_interpolated:
                continue
            bx, by, bw, bh = b.bbox
            if bx <= x <= bx+bw and by <= y <= by+bh:
                return b
        return None


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")
        self.bridge = CvBridge()
        self.get_logger().info("🤖 Smart Tower Builder 초기화 중...")
        
        self.intrinsics = None
        self.depth_scale = 0.001
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None
        
        self.detector = BlockDetector()
        self.detected_blocks = []
        
        self.robot_state = RobotState.IDLE
        self.selected_block = None
        self.target_position = None
        
        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw'
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw'
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info'
        )
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)
        
        self.gripper = None
        try:
            from DSR_ROBOT2 import wait
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            self.get_logger().info("Waiting for service /dsr01/drl/drl_start...")
            wait(2)
            if not self.gripper.initialize():
                self.get_logger().error("❌ Gripper initialization failed")
                raise Exception("Gripper initialization failed")
            self.get_logger().info("✅ 그리퍼 활성화 완료")
            
            self.gripper.move(50)
            wait(1.0)
            print("🔧 그리퍼 초기화: 50 (완전 열림)")
            
        except Exception as e:
            self.get_logger().error(f"그리퍼 설정 오류: {e}")
        
        self.get_logger().info("✅ 초기화 완료!")
    
    def synced_callback(self, color_msg, depth_msg, info_msg):
        try:
            self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge 오류: {e}")
            return
        
        if self.intrinsics is None:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width = info_msg.width
            self.intrinsics.height = info_msg.height
            self.intrinsics.ppx = info_msg.k[2]
            self.intrinsics.ppy = info_msg.k[5]
            self.intrinsics.fx = info_msg.k[0]
            self.intrinsics.fy = info_msg.k[4]
            
            if info_msg.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs.distortion.brown_conrady
            else:
                self.intrinsics.model = rs.distortion.none
            
            self.intrinsics.coeffs = list(info_msg.d)
            self.depth_scale = 0.001
            
            self.get_logger().info("✅ 카메라 파라미터 수신 완료")
            print(f"📸 depth_scale: {self.depth_scale}")
        
        if self.latest_cv_color is not None and self.latest_cv_depth_mm is not None:
            self.detected_blocks = self.detector.detect(
                self.latest_cv_color,
                self.latest_cv_depth_mm,
                self.intrinsics,
                self.depth_scale
            )
    
    def mouse_callback(self, event, u, v, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        if self.latest_cv_depth_mm is None or self.intrinsics is None:
            self.get_logger().warn("⚠️ 카메라 데이터 대기 중...")
            return
        
        if self.robot_state == RobotState.IDLE:
            block = self.detector.find_block_at(self.detected_blocks, u, v)
            
            if block and block.center_3d:
                self.selected_block = block
                self.robot_state = RobotState.BLOCK_SELECTED
                
                print("\n" + "="*60)
                print("🎯 블록 선택됨!")
                print("="*60)
                cx, cy = block.center_2d
                print(f"  📍 중심: ({cx}, {cy})")
                print(f"  📏 크기: {block.real_width_mm:.1f}x{block.real_height_mm:.1f}mm")
                print(f"  🏷️ 분류: {block.size_class}")
                print(f"  📐 깊이: {block.depth*1000:.1f}mm")
                print(f"\n👉 이제 내려놓을 위치를 클릭하세요!")
                print("="*60 + "\n")
            else:
                print("⚠️ 블록을 클릭해주세요!")
        
        elif self.robot_state == RobotState.BLOCK_SELECTED:
            try:
                depth_mm = self.latest_cv_depth_mm[v, u]
            except IndexError:
                self.get_logger().warn(f"좌표 범위 초과: ({u}, {v})")
                return
            
            if depth_mm == 0:
                print("⚠️ 깊이를 측정할 수 없습니다. 다시 클릭해주세요.")
                return
            
            depth_m = float(depth_mm) * self.depth_scale
            point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth_m)
            
            self.target_position = point_3d
            
            print("\n" + "="*60)
            print("✅ 목표 위치 선택됨!")
            print("="*60)
            print(f"  📍 픽셀: ({u}, {v})")
            print(f"  📐 3D: X={point_3d[0]*1000:.1f}, Y={point_3d[1]*1000:.1f}, Z={depth_m*1000:.1f}mm")
            print(f"\n🤖 로봇 작동 시작...")
            print("="*60 + "\n")
            
            self.execute_pick_and_place()
    
    def execute_pick_and_place(self):
        if not self.selected_block or not self.target_position:
            return
        
        try:
            from DSR_ROBOT2 import get_current_posx, movel, wait, movej
            from DR_common2 import posx, posj
            
            # 픽 좌표
            pick_rs_x = self.selected_block.center_3d[0] * 1000
            pick_rs_y = self.selected_block.center_3d[1] * 1000
            pick_rs_z = self.selected_block.depth * 1000
            
            pick_x = 685 + pick_rs_y
            pick_y = pick_rs_x + 20
            pick_z = max(811 - pick_rs_z, 150)
            
            # 플레이스 좌표
            place_rs_x = self.target_position[0] * 1000
            place_rs_y = self.target_position[1] * 1000
            place_rs_z = self.target_position[2] * 1000
            
            place_x = 685 + place_rs_y
            place_y = place_rs_x + 20
            place_z = max(811 - place_rs_z, 150)
            
            print(f"🎯 픽 위치: X={pick_x:.1f}, Y={pick_y:.1f}, Z={pick_z:.1f}")
            print(f"🎯 플레이스 위치: X={place_x:.1f}, Y={place_y:.1f}, Z={place_z:.1f}\n")
            
            current_pos = get_current_posx()[0]
            cur_Rx, cur_Ry, cur_Rz = current_pos[3], current_pos[4], current_pos[5]
            
            approach_lift = 50.0
            
            # === PICK ===
            print("📌 1단계: 블록 접근...")
            pick_up = [pick_x, pick_y, pick_z + approach_lift, cur_Rx, cur_Ry, cur_Rz]
            movel(posx(pick_up), vel=VELOCITY, acc=ACC)
            wait(0.5)
            
            print("📌 2단계: 블록 하강...")
            pick_at = [pick_x, pick_y, pick_z, cur_Rx, cur_Ry, cur_Rz]
            movel(posx(pick_at), vel=VELOCITY, acc=ACC)
            wait(0.5)
            
            print("📌 3단계: 그리퍼 닫기 (집기)... [800]")
            self.gripper.move(800)
            wait(2.0)
            
            print("📌 4단계: 블록 들어올리기...")
            movel(posx(pick_up), vel=VELOCITY, acc=ACC)
            wait(0.5)
            
            # === PLACE ===
            print("📌 5단계: 목표 위치 접근...")
            place_up = [place_x, place_y, place_z + approach_lift, cur_Rx, cur_Ry, cur_Rz]
            movel(posx(place_up), vel=VELOCITY, acc=ACC)
            wait(0.5)
            
            print("📌 6단계: 목표 위치 하강...")
            place_at = [place_x, place_y, place_z, cur_Rx, cur_Ry, cur_Rz]
            movel(posx(place_at), vel=VELOCITY, acc=ACC)
            wait(0.5)
            
            print("📌 7단계: 그리퍼 열기 (놓기)... [50]")
            self.gripper.move(50)
            wait(1.5)
            
            print("📌 8단계: 상승...")
            movel(posx(place_up), vel=VELOCITY, acc=ACC)
            wait(0.5)
            
            print("📌 9단계: 홈 복귀...")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, VELOCITY, ACC)
            wait(1.0)
            
            print("\n✅ 픽앤플레이스 완료!\n")
            
        except Exception as e:
            self.get_logger().error(f"❌ 로봇 제어 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.selected_block = None
            self.target_position = None
            self.robot_state = RobotState.IDLE
            print("💡 다음 블록을 선택해주세요!\n")
    
    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()


def main(args=None):
    rclpy.init(args=args)
    
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    
    try:
        from DSR_ROBOT2 import get_current_posx, movel, wait, movej
        from DR_common2 import posx, posj
    except ImportError as e:
        print(f"❌ DSR_ROBOT2 라이브러리 임포트 실패: {e}")
        rclpy.shutdown()
        exit(1)
    
    robot_controller = RobotControllerNode()
    
    cv2.namedWindow("Smart Tower Builder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Tower Builder", 640, 480)
    cv2.setMouseCallback("Smart Tower Builder", robot_controller.mouse_callback)
    
    # 🔥 Binary 디버그 창 추가
    cv2.namedWindow("Binary Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Binary Debug", 350, 250)
    
    print("\n" + "="*70)
    print("🏗️  SMART TOWER BUILDER - 수정 버전 v2")
    print("="*70)
    print("\n🔧 수정 사항:")
    print("  - Binary 디버그 창 추가")
    print("  - depth 필터링: 300mm ~ 1200mm만 허용")
    print("  - 배경/원거리 객체 자동 제거")
    print("  - 상세 디버깅 출력")
    print("\n📋 사용법:")
    print("  1️⃣  첫 번째 클릭: 집을 블록 선택")
    print("  2️⃣  두 번째 클릭: 내려놓을 위치 선택")
    print("  ❌ ESC 키: 종료")
    print("\n" + "="*70 + "\n")
    
    try:
        while rclpy.ok():
            rclpy.spin_once(robot_controller, timeout_sec=0.001)
            rclpy.spin_once(dsr_node, timeout_sec=0.001)
            
            if robot_controller.latest_cv_color is not None:
                display = robot_controller.latest_cv_color.copy()
                h, w, _ = display.shape
                
                d = robot_controller.detector
                
                # ROI 표시
                cv2.rectangle(display, (d.roi_x, d.roi_y),
                            (d.roi_x + d.roi_w, d.roi_y + d.roi_h),
                            (0, 255, 0), 2)
                cv2.putText(display, "ROI", (d.roi_x, d.roi_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                for block in robot_controller.detected_blocks:
                    if block.is_interpolated:
                        continue
                    
                    is_selected = (block == robot_controller.selected_block)
                    is_stable = (block.stability_score >= d.stability_threshold)
                    
                    if is_selected:
                        color = (0, 255, 255)  # 노랑
                        thickness = 4
                    elif is_stable:
                        color = (0, 255, 0)    # 초록
                        thickness = 2
                    else:
                        color = (0, 140, 255)  # 주황
                        thickness = 2
                    
                    cv2.drawContours(display, [block.rotated_box], 0, color, thickness)
                    
                    cx, cy = block.center_2d
                    cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                    
                    if block.depth > 0:
                        txt = f"{block.size_class} [{block.stability_score}]"
                        cv2.putText(display, txt, (cx - 30, cy - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 상태 표시
                status_text = {
                    RobotState.IDLE: "대기: 블록을 선택하세요",
                    RobotState.BLOCK_SELECTED: "블록 선택됨: 목표 위치 클릭",
                    RobotState.PICKING: "픽 실행 중...",
                    RobotState.PLACING: "플레이스 실행 중..."
                }
                
                status = status_text[robot_controller.robot_state]
                cv2.putText(display, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                stable_count = len([b for b in robot_controller.detected_blocks 
                                  if b.stability_score >= d.stability_threshold and not b.is_interpolated])
                cv2.putText(display, f"Blocks: {stable_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # depth 범위 표시
                depth_info = f"Depth filter: {int(d.min_depth*1000)}~{int(d.max_depth*1000)}mm"
                cv2.putText(display, depth_info, (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("Smart Tower Builder", display)
                
                # 🔥 Binary 창 표시
                if robot_controller.detector.binary_view is not None:
                    cv2.imshow("Binary Debug", robot_controller.detector.binary_view)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    except KeyboardInterrupt:
        print("\n⚠️ Ctrl+C로 종료...")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 프로그램 종료 중...")
        robot_controller.terminate_gripper()
        cv2.destroyAllWindows()
        robot_controller.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()
        print("✅ 종료 완료\n")


if __name__ == '__main__':
    main()
