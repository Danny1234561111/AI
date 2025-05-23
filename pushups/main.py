import math
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
from pathlib import Path
import cv2
import time

def angle(a, b, c):
    d = math.atan2(c[1] - b[1], c[0] - b[0])
    e = math.atan2(a[1] - b[1], a[0] - b[0])
    angle_ = np.rad2deg(d - e)
    angle_ = angle_ + 360 if angle_ < 0 else angle_
    return 360 - angle_ if angle_ > 180 else angle_
def process(image, keypoints):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    try:
        if right_elbow[0] > 0 and right_wrist[0] > 0 and right_shoulder[0] > 0:
            angle_elbow = angle(right_shoulder, right_elbow, right_wrist)
            x, y = int(right_elbow[0]) + 10, int(right_elbow[1]) + 10
        else:
            angle_elbow = angle(left_shoulder, left_elbow, left_wrist)
            x, y = int(left_elbow[0]) + 10, int(left_elbow[1]) + 10

        cv2.putText(image, f"{int(angle_elbow)}", (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 25, 255), 2)
        return angle_elbow
    except ZeroDivisionError:
        pass
    return None

path = Path(__file__).parent
model_path = path / "yolo11n-pose.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture('press_kachat.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_file = 'res.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

flag = False
count = 0
last_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    result = results[0]

    keypoints = result.keypoints.xy.tolist()
    if not keypoints or not keypoints[0]:
        continue

    keypoints = keypoints[0]

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()

    angle_ = process(annotated, keypoints)

    if angle_ is not None:
        if flag and angle_ > 150:
            count += 1
            flag = False
            last_time = time.time()
        elif angle_ < 90:
            flag = True

    if time.time() - last_time > 15:
        count = 0

    cv2.putText(annotated, f"Push-ups: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(annotated)
    cv2.imshow("Push-up Counter", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()
