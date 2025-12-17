# 🏗️ RealSense Block Detection (v5)

Intel RealSense 뎁스 카메라와 OpenCV를 활용하여 환경 내의 블록을 고속으로 감지하고, 그 **3D 위치 및 실제 크기**를 정확하게 측정하는 Python 프로젝트입니다.

## ✨ 주요 특징

* **RealSense 뎁스 이미지 직접 사용**: 뎁스 이미지(Numpy Array)에서 뎁스 값을 직접 읽고 처리하여, 빠르고 견고한 3D 측정을 수행합니다.
* **견고한 뎁스 측정**: 중심 픽셀의 뎁스가 불안정할 경우, 주변 픽셀을 샘플링하여 뎁스 중간값을 사용하여 정확도를 높입니다.
* **실시간 3D 좌표 및 크기 계산**: 픽셀 좌표를 카메라 고유 정보(**Intrinsics**)를 활용하여 월드 좌표계(X, Y, Z)로 변환하고, 블록의 실제 크기(mm)를 계산합니다.
    * $$\text{Real Size} = \frac{\text{Pixel Size} \times \text{Depth} \times 1000}{\text{Focal Length (pixels)}}$$
* **회전 감지**: `cv2.minAreaRect`를 사용하여 블록의 회전 각도를 측정합니다.
* **실시간 튜닝**: OpenCV 트랙바를 통해 임계값(Threshold) 및 면적 필터링을 실시간으로 조정 가능합니다.

## 🛠️ 환경 및 설치

### 요구사항

* **하드웨어**: Intel RealSense 뎁스 카메라 (D400 시리즈 권장)
* **소프트웨어**:
    * Python 3.x
    * `pyrealsense2` (버전: 2.56.5.9235)
    * `numpy` (버전: 2.2.6)
    * `opencv-python` (버전: 4.12.0)

### 설치

```bash
# 필요한 라이브러리 설치
pip install pyrealsense2 numpy opencv-python