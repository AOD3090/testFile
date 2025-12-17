import cv2
import numpy as np

def nothing(x):
    pass

# 0️⃣ 카메라 열기
cap = cv2.VideoCapture(6)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

# 🎚️ 조절바 윈도우 생성
cv2.namedWindow("Control")
cv2.createTrackbar("Threshold", "Control", 200, 255, nothing)

# 🔴 ROI 설정
roi_x = 190
roi_y = 140
roi_w = 230
roi_h = 180

# 🔥 추가: 이전 중심점 저장 변수 (처음엔 None)
prev_cx = None
prev_cy = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI 사각형 표시
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 3)
    cv2.putText(frame, "ROI Area", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ROI crop
    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_val = cv2.getTrackbarPos("Threshold", "Control")
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 이번 프레임에서 검출된 모든 중심점을 임시로 저장 (여러 개일 수 있음)
    current_centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if 100 < area < 3000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            box[:, 0] += roi_x
            box[:, 1] += roi_y
            
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + roi_x
                cy = int(M["m01"] / M["m00"]) + roi_y
                
                # 화면에 표시
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{cx},{cy}", (cx, cy-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                
                # 이번 프레임 중심점 리스트에 추가
                current_centers.append((cx, cy))

    # 🔥 중심점 변경 감지 및 출력 로직
    # 현재 검출된 중심점이 하나라도 있으면
    if current_centers:
        # 간단히 첫 번째 검출된 객체의 중심점만 추적 (원하시면 모두 추적 가능)
        cx, cy = current_centers[0]
        
        # 이전 값과 비교해서 하나라도 다르면 출력
        if prev_cx is None or cx != prev_cx or cy != prev_cy:
            print(f"중심점 변경: ({cx}, {cy})")
            prev_cx, prev_cy = cx, cy
    else:
        # 객체가 사라졌을 때도 알려주고 싶으면 아래 주석 해제
        # if prev_cx is not None:
        #     print("객체 사라짐")
        #     prev_cx, prev_cy = None, None
        prev_cx, prev_cy = None, None  # 객체 없을 때 초기화

    # 화면 출력
    cv2.imshow("Result (with ROI)", frame)
    cv2.imshow("Binary View (ROI only)", binary)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()