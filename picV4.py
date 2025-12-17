import cv2
import numpy as np

def nothing(x):
    pass

# ===============================
# 카메라 열기
# ===============================
cap = cv2.VideoCapture(6)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

# ===============================
# 트랙바
# ===============================
cv2.namedWindow("Control")
cv2.createTrackbar("Threshold", "Control", 200, 255, nothing)

# ===============================
# ROI 설정 (기존 그대로)
# ===============================
roi_x = 190
roi_y = 140
roi_w = 230
roi_h = 180

# ===============================
# 메인 루프
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI 표시
    cv2.rectangle(frame, (roi_x, roi_y),
                  (roi_x + roi_w, roi_y + roi_h),
                  (0, 0, 255), 3)

    roi_frame = frame[roi_y:roi_y + roi_h,
                      roi_x:roi_x + roi_w]

    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_val = cv2.getTrackbarPos("Threshold", "Control")
    _, binary = cv2.threshold(
        blurred, thresh_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_id = 0  # 사각형 번호

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 3000:

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # 꼭짓점 4개 → 사각형
            if len(approx) == 4:
                rect_id += 1
                corners = []

                for p in approx:
                    x = p[0][0] + roi_x
                    y = p[0][1] + roi_y
                    corners.append((x, y))
                    cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

                cv2.drawContours(
                    frame, [np.array(corners)], -1, (0, 255, 0), 2)

                # 중심점
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + roi_x
                    cy = int(M["m01"] / M["m00"]) + roi_y
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # 출력
                print(f"사각형 #{rect_id}")
                print(f"중심 좌표: ({cx}, {cy})")
                print("꼭짓점 좌표 (픽셀)")
                for i, (x, y) in enumerate(corners):
                    print(f"{i+1}: ({x}, {y})")
                print("-" * 40)

    cv2.imshow("Result (with ROI)", frame)
    cv2.imshow("Binary (ROI)", binary)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
