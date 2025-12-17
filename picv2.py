import cv2
import numpy as np

def nothing(x):
    pass

# 0️⃣ 카메라 열기 (번호 안 맞으면 0,1,2... 순서로 바꿔보세요)
cap = cv2.VideoCapture(6)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

# 🎚️ 조절바 윈도우 생성
cv2.namedWindow("Control")
cv2.createTrackbar("Threshold", "Control", 200, 255, nothing)

# 🔴 ROI (관심 영역) 사각형 좌표 설정 - 여기만 자유롭게 수정하세요!
# 형식: (x, y, width, height)  ← 카메라 화면에서 원하는 영역
roi_x = 190      # 왼쪽 시작점
roi_y = 90       # 위쪽 시작점
roi_w = 250      # 너비
roi_h = 250      # 높이
# 예: 화면 중앙 영역으로 설정한 값 (640x480 기준으로 적당히 맞춤)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔴 ROI 사각형을 원본 화면에 빨간색으로 표시 (시각적으로 확인용)
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 3)
    cv2.putText(frame, "ROI Area", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 🔴 ROI 영역만 잘라내기 (crop)
    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # 1️⃣ ROI 안에서만 처리 시작
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2️⃣ 트랙바로 실시간 이진화 조절
    thresh_val = cv2.getTrackbarPos("Threshold", "Control")
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    # 3️⃣ 윤곽선 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 🎯 크기 필터링 (노이즈 제거)
        if 115 < area < 3000:
            # 최소 면적 회전 사각형
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 🔴 중요: ROI 좌표 보정 (원본 프레임 기준으로 변환)
            box[:, 0] += roi_x   # x 좌표에 roi_x 더하기
            box[:, 1] += roi_y   # y 좌표에 roi_y 더하기
            
            # 네모 그리기
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            # 중심점 계산 및 표시 (보정된 좌표)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + roi_x
                cy = int(M["m01"] / M["m00"]) + roi_y
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{cx},{cy}", (cx, cy-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # 화면 출력
    cv2.imshow("Result (with ROI)", frame)        # ROI 사각형 + 검출 결과
    cv2.imshow("Binary View (ROI only)", binary)  # ROI 내부 이진화 화면

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()