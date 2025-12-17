import cv2
import numpy as np

def nothing(x):
    pass

# 0️⃣ 카메라 열기
cap = cv2.VideoCapture(6) # 안되면 0, 1, 2 등으로 변경
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

# 🎚️ 조절바 윈도우 생성
cv2.namedWindow("Control")
cv2.createTrackbar("Threshold", "Control", 200, 255, nothing) # 초기값 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ 흑백 변환 및 블러링 (노이즈 제거)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2️⃣ 트랙바 값으로 이진화 (실시간 조절 가능)
    thresh_val = cv2.getTrackbarPos("Threshold", "Control")
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    # 3️⃣ 윤곽선 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 🎯 크기 필터링 (중요!)
        # 큐브보다 작거나(노이즈), 기계처럼 너무 큰 것(>5000)은 무시
        if 500 < area < 5000:
            
            # 모양 상관없이 사각형 박스 씌우기
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 네모 그리기
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            # 중심점 그리기
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{cx},{cy}", (cx, cy-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # 화면 출력
    cv2.imshow("Result", frame)
    cv2.imshow("Binary View", binary) # 흑백 화면을 보면서 조절하세요

    if cv2.waitKey(1) & 0xFF == 27: # ESC로 종료
        break

cap.release()
cv2.destroyAllWindows()