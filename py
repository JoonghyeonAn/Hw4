Step 1. 영상 파일 업로드
from google.colab import files
import os

# 업로드 디렉토리 생성
upload_dir = "/content/uploads"
os.makedirs(upload_dir, exist_ok=True)

# 영상 파일 업로드
uploaded = files.upload()

# 업로드된 파일 경로 설정
uploaded_filename = list(uploaded.keys())[0]
video_path = os.path.join(upload_dir, uploaded_filename)

print(f"✅ 업로드 완료: {video_path}")

Step 2. 필수 라이브러리 설치 (YOLOv5 포함)
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt
%cd /content

Step 3. YOLOv5 모델 로드 및 추론 준비
import torch
import cv2
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox

model = DetectMultiBackend("yolov5s.pt", device='cpu')
stride, names = model.stride, model.names

Step 4. 트래킹 + 속도 계산 + 선형 경로 예측 + 충돌 예측 + 로그 저장
# 초기 설정
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = "/content/output_tracking.mp4"
log_path = "/content/collision_log.txt"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

prev_positions = {}
log_file = open(log_path, "w")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = letterbox(frame, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=[2, 3, 5, 7])  # 차량 관련 class

    results = pred[0]
    boxes = results[:, :4]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 차량 ID: 여기선 임시로 i로 사용 (정식 트래커를 쓰면 개선 가능)
        vehicle_id = i

        # 속도 계산
        if vehicle_id in prev_positions:
            px, py = prev_positions[vehicle_id]
            dx = cx - px
            dy = cy - py
            distance = np.sqrt(dx**2 + dy**2)
            speed = distance * fps  # 단위: pixel/frame → pixel/sec

            # 방향
            direction = np.arctan2(dy, dx)

            # 선형 경로 예측
            future_pos = (cx + dx, cy + dy)

            # 충돌 시간 예측
            for vid2, (px2, py2) in prev_positions.items():
                if vid2 == vehicle_id:
                    continue
                dist = np.sqrt((future_pos[0]-px2)**2 + (future_pos[1]-py2)**2)
                relative_speed = abs(speed - np.linalg.norm([px2-cx, py2-cy])*fps)
                if relative_speed > 0:
                    time_to_collision = dist / relative_speed
                    if time_to_collision < 2.0:
                        msg = f"[⚠️ 충돌 예상] 차량 {vehicle_id} vs 차량 {vid2} : {time_to_collision:.2f}초 후"
                        log_file.write(msg + "\n")
                        print(msg)

        prev_positions[vehicle_id] = (cx, cy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{vehicle_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    out.write(frame)

cap.release()
if out:
    out.release()
log_file.close()

print("🎬 완료된 영상 저장: ", output_path)
print("📄 충돌 로그 확인: ", log_path)
