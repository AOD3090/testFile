"""
🏗️ Block Detection - RealSense (v5)
====================================
- depth_image (numpy)에서 직접 뎁스 읽기
- get_distance() 대신 배열 접근
- 밝은 모드(THRESH_BINARY)만 사용하도록 수정
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


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
    angle: float = 0.0  # 새로 추가: 회전 각도


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.depth_scale = 0.001  # 기본값 (보통 0.001 = 1mm)
        
    def start(self) -> bool:
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            
            # 🔥 depth_scale 가져오기
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"   Depth Scale: {self.depth_scale} (1 unit = {self.depth_scale*1000:.2f}mm)")
            
            depth_stream = profile.get_stream(rs.stream.depth)
            self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            print("📷 카메라 초기화 중...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
            print("✅ RealSense 시작!")
            print(f"   해상도: {self.width}x{self.height}")
            return True
        except Exception as e:
            print(f"❌ 카메라 오류: {e}")
            return False
    
    def get_frames(self):
        """필터 없이 원본 프레임 반환"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # uint16, raw값
            
            return color_image, depth_image, depth_frame
        except:
            return None, None, None
    
    def get_depth_from_image(self, x, y, depth_image, debug=False) -> float:
        """
        🔥 depth_image (numpy array)에서 직접 뎁스 읽기
        depth_image는 uint16, 값 * depth_scale = 미터
        """
        x = int(x)
        y = int(y)
        
        if not (0 <= x < self.width and 0 <= y < self.height):
            if debug:
                print(f"   [DEBUG] 좌표 범위 초과: ({x}, {y})")
            return 0.0
        
        # 중심점 값
        raw_depth = depth_image[y, x]
        depth_m = raw_depth * self.depth_scale
        
        if debug:
            print(f"   [DEBUG] 중심점 ({x},{y}): raw={raw_depth}, {depth_m*100:.1f}cm")
        
        if depth_m > 0.05:  # 5cm 이상이면 유효
            return depth_m
        
        # 중심이 0이면 주변 샘플링
        sample_offsets = [
            (-5, 0), (5, 0), (0, -5), (0, 5),
            (-10, 0), (10, 0), (0, -10), (0, 10),
            (-5, -5), (5, -5), (-5, 5), (5, 5),
            (-15, 0), (15, 0), (0, -15), (0, 15),
        ]
        
        valid_depths = []
        for dx, dy in sample_offsets:
            sx, sy = x + dx, y + dy
            if 0 <= sx < self.width and 0 <= sy < self.height:
                raw = depth_image[sy, sx]
                d = raw * self.depth_scale
                if 0.05 < d < 3.0:
                    valid_depths.append(d)
        
        if debug:
            print(f"   [DEBUG] 주변 샘플링: {len(valid_depths)}개 유효")
            if valid_depths:
                print(f"   [DEBUG] 범위: {min(valid_depths)*100:.1f}cm ~ {max(valid_depths)*100:.1f}cm")
        
        if valid_depths:
            valid_depths.sort()
            return valid_depths[len(valid_depths) // 2]
        
        return 0.0
    
    def pixel_to_3d_from_image(self, x, y, depth_image, debug=False):
        """depth_image 사용하여 3D 좌표 계산"""
        depth = self.get_depth_from_image(x, y, depth_image, debug=debug)
        
        if depth <= 0:
            return None
        
        # 3D 변환
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
        return (point[0], point[1], depth)
    
    def calc_real_size(self, w_px, h_px, depth):
        if depth <= 0 or not self.intrinsics:
            return 0, 0
        real_w = (w_px * depth * 1000) / self.intrinsics.fx
        real_h = (h_px * depth * 1000) / self.intrinsics.fy
        return real_w, real_h
    
    def stop(self):
        if self.pipeline:
            self.pipeline.stop()


class BlockDetector:
    def __init__(self):
        self.threshold = 200
        self.use_inv = True  # 밝은 모드만 사용 (THRESH_BINARY)
        
        self.min_area = 90
        self.max_area = 4000
        self.roi_x = 190
        self.roi_y = 140
        self.roi_w = 230
        self.roi_h = 180
        
        self.min_aspect = 0.5
        self.max_aspect = 3.0
        
        self.min_solidity = 0.7
        
        self.min_depth = 0.1
        self.max_depth = 2.0  
        
        self.small_thresh = 30
        self.medium_thresh = 50
        
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
            if not (self.min_area < area < self.max_area):
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            (_, _), (w, h), angle = rect  # 각도 추출
            if h == 0 or w == 0:
                continue
            aspect = max(w, h) / min(w, h)
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < self.min_solidity:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if not (3 <= len(approx) <= 8):
                continue
            
            box_global = box.copy()
            box_global[:, 0] += self.roi_x
            box_global[:, 1] += self.roi_y
            cnt_global = cnt.copy()
            cnt_global[:, :, 0] += self.roi_x
            cnt_global[:, :, 1] += self.roi_y
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"]) + self.roi_x
            cy = int(M["m01"] / M["m00"]) + self.roi_y
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            block = Block(
                bbox=(x + self.roi_x, y + self.roi_y, bw, bh),
                center_2d=(cx, cy),
                contour=cnt_global,
                rotated_box=box_global,
                area=area,
                aspect_ratio=aspect,
                solidity=solidity,
                angle=angle  # 각도 저장
            )
            
            # 🔥 depth_image에서 직접 읽기
            point_3d = camera.pixel_to_3d_from_image(cx, cy, depth_image, debug=False)
            
            if point_3d:
                block.center_3d = point_3d
                block.depth = point_3d[2]
                
                if self.min_depth < block.depth < self.max_depth:
                    real_w, real_h = camera.calc_real_size(bw, bh, block.depth)
                    block.real_width_mm = real_w
                    block.real_height_mm = real_h

            
            blocks.append(block)
        return blocks
    
    def find_at(self, blocks, x, y):
        for b in blocks:
            bx, by, bw, bh = b.bbox
            if bx <= x <= bx+bw and by <= y <= by+bh:
                return b
        return None


# 전역 변수
g_blocks = []
g_detector = None
g_camera = None
g_depth_image = None
g_selected_idx = -1


def mouse_callback(event, x, y, flags, param):
    global g_blocks, g_detector, g_camera, g_depth_image, g_selected_idx
    
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    
    if g_detector is None or g_camera is None or g_depth_image is None:
        return
    
    for i, block in enumerate(g_blocks):
        bx, by, bw, bh = block.bbox
        if bx <= x <= bx+bw and by <= y <= by+bh:
            g_selected_idx = i
            
            print("\n" + "="*50)
            print("🎯 블록 정보")
            print("="*50)
            cx, cy = block.center_2d
            print(f"  📍 중심점 (픽셀): ({cx}, {cy})")
            
            # 디버깅 출력
            print(f"\n  [뎁스 디버깅]")
            point_3d = g_camera.pixel_to_3d_from_image(cx, cy, g_depth_image, debug=True)
            
            if point_3d and point_3d[2] > 0:
                X, Y, Z = point_3d
                depth = Z
                
                # 🔥 변경: minAreaRect의 w, h 사용 (detect에서 추출된 rect 사용 위해 block에 저장되지 않았으므로, 여기서 재계산하거나 block에 추가 필요)
                # block.angle이 이미 rect에서 왔으므로, detect에서 w, h도 block에 저장하도록 수정 추천. 하지만 "나머지 가만히"라 여기서 cnt로 재계산.
                # 실제로는 detect에서 block에 w, h 추가하는 게 좋지만, 최소 변경으로: 여기서 rect 재계산.
                rect = cv2.minAreaRect(block.contour)
                (_, _), (w, h), _ = rect  # minAreaRect의 w, h 추출
                
                print(f"\n  3D 좌표: X={X*1000:.1f}mm, Y={Y*1000:.1f}mm, Z={depth*1000:.1f}mm")
                real_w, real_h = g_camera.calc_real_size(w, h, depth)  # 🔥 변경: bw, bh 대신 w, h 사용
                print(f"  블록 크기: {real_w:.1f} x {real_h:.1f} mm")
                
                side_length = (real_w + real_h) / 2
                print(f"  한 변 길이: {side_length:.1f} mm")
                print(f"  각도: {block.angle:.1f}°")  # 각도 출력 추가
                
            else:
                print("\n  ⚠️ 뎁스 정보 없음")
            
            print("="*50 + "\n")
            return
    
    g_selected_idx = -1
    
    # 빈 공간 클릭
    print(f"\n빈 공간 클릭 ({x}, {y})")
    d = g_camera.get_depth_from_image(x, y, g_depth_image, debug=True)
    
    if d > 0:
        print(f"  → 해당 위치 뎁스: {d*100:.1f}cm")

        X, Y, Z = rs.rs2_deproject_pixel_to_point(
            g_camera.intrinsics, [x, y], d
        )

        print(
            f"  3D 좌표: X={X*1000:.1f}mm, "
            f"Y={Y*1000:.1f}mm, "
            f"Z={Z*1000:.1f}mm"
        )
    
    else:
        print(f"  → 뎁스 측정 불가\n")


def main():
    global g_blocks, g_detector, g_camera, g_depth_image, g_selected_idx
    
    print("\n" + "="*50)
    print("🏗️ Block Detection - RealSense (v5)")
    print("   depth_image 직접 읽기 버전   ")
    print("="*50)
    
    g_camera = RealSenseCamera()
    if not g_camera.start():
        return
    
    g_detector = BlockDetector()
    
    cv2.namedWindow("Result")
    cv2.setMouseCallback("Result", mouse_callback)
    
    cv2.namedWindow("Control")
    cv2.createTrackbar("Threshold", "Control", g_detector.threshold, 255, lambda x: None)
    cv2.createTrackbar("Min Area", "Control", g_detector.min_area, 5000, lambda x: None)
    cv2.createTrackbar("Max Area", "Control", g_detector.max_area, 30000, lambda x: None)
    # White/Black 트랙바 제거 (밝은 모드 고정)
    
    print("\n📌 조작:")
    print("   - 블록 클릭: 정보 + 뎁스 디버깅")
    print("   - 빈 공간 클릭: 해당 위치 뎁스 확인")
    print("   - ESC: 종료")
    print("="*50 + "\n")
    
    try:
        while True:
            g_detector.threshold = cv2.getTrackbarPos("Threshold", "Control")
            g_detector.min_area = cv2.getTrackbarPos("Min Area", "Control")
            g_detector.max_area = cv2.getTrackbarPos("Max Area", "Control")
            
            color, depth_img, depth_frame = g_camera.get_frames()
            if color is None:
                continue
            
            g_depth_image = depth_img  # 전역에 저장
            
            g_blocks = g_detector.detect(color, depth_img, g_camera)
            
            # 그리기
            display = color.copy()
            d = g_detector
            
            cv2.rectangle(display, (d.roi_x, d.roi_y), 
                         (d.roi_x+d.roi_w, d.roi_y+d.roi_h), (0, 0, 255), 2)
            mode_text = "WHITE"  # 밝은 모드 고정
            cv2.putText(display, f"ROI ({mode_text})", (d.roi_x, d.roi_y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            for i, block in enumerate(g_blocks):
                is_sel = (i == g_selected_idx)
                color_draw = (0, 255, 255) if is_sel else (0, 255, 0)
                thickness = 3 if is_sel else 2
                
                cv2.drawContours(display, [block.rotated_box], 0, color_draw, thickness)
                cx, cy = block.center_2d
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                
                if block.depth > 0:
                    # 1줄: 거리
                    txt_depth = f"{block.depth*100:.0f}cm"
                    cv2.putText(display, txt_depth, (cx-15, cy-25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_draw, 2)
                    
                    # 2줄: 가로 x 세로 (mm)
                    txt_size = f"{block.real_width_mm:.0f}x{block.real_height_mm:.0f}mm"
                    cv2.putText(display, txt_size, (cx-30, cy-8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                else:
                    cv2.putText(display, "no depth", (cx-25, cy-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                cv2.putText(display, f"({cx},{cy})", (cx-25, cy+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(display, f"Blocks: {len(g_blocks)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Result", display)
            
            if g_detector.binary_view is not None:
                cv2.imshow("Binary (ROI)", g_detector.binary_view)
            
            # Depth 시각화 (값 범위 확인용)
            depth_display = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # 뎁스 값 범위 표시
            valid_mask = depth_img > 0
            if np.any(valid_mask):
                min_d = np.min(depth_img[valid_mask]) * g_camera.depth_scale * 100
                max_d = np.max(depth_img[valid_mask]) * g_camera.depth_scale * 100
                cv2.putText(depth_display, f"Range: {min_d:.0f}-{max_d:.0f}cm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Depth", depth_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        g_camera.stop()
        cv2.destroyAllWindows()
        print("👋 종료")


if __name__ == "__main__":
    main()