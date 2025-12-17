import cv2
import numpy as np

# 0️⃣ 카메라 열기
cap = cv2.VideoCapture(6)  # 필요하면 번호 변경
if not cap.isOpened():
    raise RuntimeError("카메라 열기 실패")

while True:
    # 1️⃣ 프레임 읽기
    ret, img = cap.read()
    if not ret:
        break

    # 2️⃣ 흑백 → 이진 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 표시용 이미지
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # 3️⃣ 윤곽선 검출
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        # 4️⃣ 네모 근사
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) != 4:
            continue

        # ---------- 크기 특정 시작 ----------

        area = cv2.contourArea(approx)
        if not (2000 <= area <= 50000):
            continue

        pts = approx.reshape(4, 2)
        lengths = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            lengths.append(np.linalg.norm(p1 - p2))

        min_len = min(lengths)
        max_len = max(lengths)
        if not (30 <= min_len and max_len <= 300):
            continue

        # ---------- 크기 특정 끝 ----------

        # 네모 그리기
        cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)

        # 5️⃣ 중점 계산
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1)

    # 6️⃣ 화면 표시용 사이즈 축소
    result_small = cv2.resize(result, None, fx=0.2, fy=0.2)

    cv2.imshow("result", result_small)

    # ESC 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
