import cv2
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from skimage import draw

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

model_path = "facial_best.pt"
original_oranges = cv2.imread("oranges.png")
oranges = original_oranges.copy()
hsv_oranges = cv2.cvtColor(oranges, cv2.COLOR_BGR2HSV)

lower = np.array((10, 240, 200))
upper = np.array((15, 255, 255))
mask = cv2.inRange(hsv_oranges, lower, upper)
mask = cv2.dilate(mask, np.ones((7, 7)))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_contours = sorted(contours, key=cv2.contourArea)
m = cv2.moments(sorted_contours[-1])
cx = int(m["m10"] / m["m00"])
cy = int(m["m01"] / m["m00"])

bbox = cv2.boundingRect(sorted_contours[-1])
x, y, w, h = bbox

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

model = YOLO(model_path)

while(True):
    ret, image = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    oranges = original_oranges.copy()

    result = model(image)[0]
    masks = result.masks

    if masks is not None:
        global_mask = masks.data.cpu().numpy()[0] if len(masks) > 0 else np.zeros(image.shape[:2], dtype=np.float32)

        if len(masks) > 1:
            for mask in masks.data.cpu().numpy()[1:]:
                global_mask += mask

        global_mask = cv2.resize(global_mask, (image.shape[1], image.shape[0])).astype(np.float32)

        rr, cc = draw.disk((5, 5), 5)
        struct = np.zeros((11, 11), np.uint8)
        struct[rr, cc] = 1

        global_mask = cv2.dilate(global_mask, struct, iterations=2)

        pos = np.where(global_mask > 0.5)
        if len(pos[0]) > 0:

            min_y, max_y = int(np.min(pos[0])), int(np.max(pos[0]))
            min_x, max_x = int(np.min(pos[1])), int(np.max(pos[1]))

            face = image[min_y:max_y, min_x:max_x]
            global_mask_face = global_mask[min_y:max_y, min_x:max_x]

            resized_face = cv2.resize(face, (w, h))
            resized_mask_face = cv2.resize(global_mask_face, (w, h)).astype(np.float32)

            resized_mask_face = (resized_mask_face > 0.5).astype(np.uint8) * 255

            roi = oranges[y:y + h, x:x + w]

            if roi.shape == resized_face.shape and roi.size > 0:
                roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(resized_mask_face))
                face_fg = cv2.bitwise_and(resized_face, resized_face, mask=resized_mask_face)
                combined = cv2.add(roi_bg, face_fg)
                oranges[y:y + h, x:x + w] = combined
            else:
                print("Warning: ROI and resized_face have incompatible shapes or zero size, skipping replacement.")
        else:
            print("Warning: No face detected in image.")

    cv2.imshow("Image", oranges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
