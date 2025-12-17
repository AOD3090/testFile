"""
🏗️ Block Detection App - RealSense (v5 Class Version)
======================================================
- UI, Camera, Logic을 BlockDetectionApp 클래스로 통합
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ==========================================
# 1. 데이터 클래스 및 기본 클래스 (기존 유지)
# ==========================================

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


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.depth_scale = 0.001
        
    def start(self) -> bool:
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"   Depth Scale: {self.depth_scale}")
            
            depth_stream = profile.get_stream(rs.stream.depth)
            self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            # 워밍업
            for _ in range(10):
                self.pipeline.wait_for_frames()
            
            print("✅ RealSense 시작됨")
            return True
        except Exception as e:
            print(f"❌ 카메라 오류: {e}")
            return False
    
    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image, depth_frame
        except:
            return None, None, None
    
    def get_depth_from_image(self, x, y, depth_image, debug=False) -> float:
        x, y = int(x), int(y)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        
        raw_depth = depth_image[y, x]
        depth_m = raw_depth * self.depth_scale
        
        if depth_m > 0.05:
            return depth_m
            
        # 주변 샘플링
        offsets = [(-5,0), (5,0), (0,-5), (0,5), (-2, -2), (2, 2)]
        valid = []
        for dx, dy in offsets:
            sx, sy = x+dx, y+dy
            if 0 <= sx < self.width and 0 <= sy < self.height:
                d = depth_image[sy, sx] * self.depth_scale
                if 0.05 < d < 3.0:
                    valid.append(d)
        
        if valid:
            valid.sort()
            return valid[len(valid)//2]
        return 0.0
    
    def pixel_to_3d_from_image(self, x, y, depth_image, debug=False):
        depth = self.get_depth_from_image(x, y, depth_image, debug)
        if depth <= 0: return None
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
        return (point[0], point[1], depth)
    
    def calc_real_size(self, w_px, h_px, depth):
        if depth <= 0 or not self.intrinsics: return 0, 0
        real_w = (w_px * depth * 1000) / self.intrinsics.fx
        real_h = (h_px * depth * 1000) / self.intrinsics.fy
        return real_w, real_h
    
    def stop(self):
        if self.pipeline:
            self.pipeline.stop()


class BlockDetector:
    def __init__(self):
        self.threshold = 200
        self.min_area = 90
        self.max_area = 10000
        self.roi_x = 190
        self.roi_y = 140
        self.roi_w = 230
        self.roi_h = 180
        self.min_depth = 0.1
        self.max_depth = 2.0
        self.binary_view = None
        
    def detect(self, frame, depth_image, camera) -> List[Block]:
        blocks = []
        roi = frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self.binary_view = binary
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area): continue
            
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            (_, _), (w, h), _ = rect
            
            # ROI 좌표 보정
            box_global = box.copy()
            box_global[:, 0] += self.roi_x
            box_global[:, 1] += self.roi_y
            
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"]) + self.roi_x
            cy = int(M["m01"] / M["m00"]) + self.roi_y
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            block = Block(
                bbox=(x + self.roi_x, y + self.roi_y, bw, bh),
                center_2d=(cx, cy),
                contour=cnt, # ROI 내부 컨투어
                rotated_box=box_global,
                area=area
            )
            
            # 3D 변환
            point_3d = camera.pixel_to_3d_from_image(cx, cy, depth_image)
            if point_3d:
                block.center_3d = point_3d
                block.depth = point_3d[2]
                if self.min_depth < block.depth < self.max_depth:
                    rw, rh = camera.calc_real_size(bw, bh, block.depth)
                    block.real_width_mm = rw
                    block.real_height_mm = rh
            
            blocks.append(block)
        return blocks


# ==========================================
# 2. 메인 앱 클래스 (통합 및 리팩토링)
# ==========================================

class BlockDetectionApp:
    def __init__(self):
        self.camera = RealSenseCamera()
        self.detector = BlockDetector()
        
        self.blocks = []
        self.depth_image = None
        self.selected_idx = -1
        self.is_running = False

    def init_windows(self):
        """OpenCV 윈도우 및 트랙바 설정"""
        cv2.namedWindow("Result")
        cv2.setMouseCallback("Result", self.mouse_callback)
        
        cv2.namedWindow("Control")
        cv2.createTrackbar("Threshold", "Control", self.detector.threshold, 255, lambda x: None)
        cv2.createTrackbar("Min Area", "Control", self.detector.min_area, 5000, lambda x: None)
        cv2.createTrackbar("Max Area", "Control", self.detector.max_area, 30000, lambda x: None)

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 핸들러"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        if self.depth_image is None:
            return

        # 블록 선택 로직
        self.selected_idx = -1
        for i, block in enumerate(self.blocks):
            bx, by, bw, bh = block.bbox
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.selected_idx = i
                self.print_block_info(block)
                return

        # 빈 공간 클릭
        self.print_point_depth(x, y)

    def print_block_info(self, block):
        print("\n" + "="*50)
        print("🎯 블록 정보 선택됨")
        cx, cy = block.center_2d
        
        point_3d = self.camera.pixel_to_3d_from_image(cx, cy, self.depth_image, debug=True)
        if point_3d and point_3d[2] > 0:
            X, Y, Z = point_3d
            print(f"  🌐 3D: X={X*1000:.1f}, Y={Y*1000:.1f}, Z={Z*1000:.1f}mm")
            print(f"  📏 크기: {block.real_width_mm:.1f} x {block.real_height_mm:.1f} mm")
        else:
            print("  ⚠️ 뎁스 정보 없음")
        print("="*50)

    def print_point_depth(self, x, y):
        d = self.camera.get_depth_from_image(x, y, self.depth_image, debug=True)
        print(f"\n📍 좌표({x}, {y}) 뎁스: {d*100:.1f}cm\n")

    def update_params(self):
        """트랙바 값 업데이트"""
        self.detector.threshold = cv2.getTrackbarPos("Threshold", "Control")
        self.detector.min_area = cv2.getTrackbarPos("Min Area", "Control")
        self.detector.max_area = cv2.getTrackbarPos("Max Area", "Control")

    def draw_hud(self, display):
        """화면에 텍스트 및 ROI 그리기"""
        d = self.detector
        cv2.rectangle(display, (d.roi_x, d.roi_y), (d.roi_x+d.roi_w, d.roi_y+d.roi_h), (0, 0, 255), 2)
        cv2.putText(display, f"Blocks: {len(self.blocks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for i, block in enumerate(self.blocks):
            is_sel = (i == self.selected_idx)
            color = (0, 255, 255) if is_sel else (0, 255, 0)
            thick = 3 if is_sel else 2
            
            cv2.drawContours(display, [block.rotated_box], 0, color, thick)
            cx, cy = block.center_2d
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            
            # 정보 텍스트
            if block.depth > 0:
                cv2.putText(display, f"{block.depth*100:.0f}cm", (cx-15, cy-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(display, f"{block.real_width_mm:.0f}x{block.real_height_mm:.0f}", (cx-30, cy-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def run(self):
        print("\n=== App Started (Press q to quit) ===")
        if not self.camera.start():
            return

        self.init_windows()
        self.is_running = True
        
        try:
            while self.is_running:
                self.update_params()
                
                color, depth_img, _ = self.camera.get_frames()
                if color is None: continue
                
                self.depth_image = depth_img
                self.blocks = self.detector.detect(color, depth_img, self.camera)
                
                display = color.copy()
                self.draw_hud(display)
                
                cv2.imshow("Result", display)
                if self.detector.binary_view is not None:
                    cv2.imshow("Binary", self.detector.binary_view)
                
                if cv2.waitKey(1) & 0xFF == 113: # q
                    break
                    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            print("=== App Stopped ===")

if __name__ == "__main__":
    app = BlockDetectionApp()
    app.run()