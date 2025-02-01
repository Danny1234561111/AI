import cv2
import numpy as np
import matplotlib.pyplot as plt

lbp_cascades="lbpcascades/lbpcascade_frontalface.xml"
hear_cascades="haarcascades/haarcascade_eye_tree_eyeglasses.xml"


face=cv2.CascadeClassifier(hear_cascades)
face1=cv2.CascadeClassifier(lbp_cascades)

glass = cv2.imread("1.jpeg")
gray_glass = cv2.cvtColor(glass, cv2.COLOR_BGR2GRAY)
_, binary_glass = cv2.threshold(gray_glass, 240, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary_glass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(glass)
largest_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
glass = cv2.bitwise_and(glass, mask)




def detector(img,classifier,scaleFactor=None,minNeightBours=None):
    result=img.copy()
    rects=classifier.detectMultiScale(result,scaleFactor=scaleFactor,minNeighbors=minNeightBours)
    min_x = min_y=1000000000
    max_x = max_y = 0

    # Проход по всем прямоугольникам
    for (x, y, w, h) in rects:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(x+w, max_x)
        max_y = max(y+h, max_y)

        cv2.rectangle(result,(x,y),(x+w,y+h),(255,255,255))
    print(len(rects))
    if (len(rects)>1):
        w = max_x - min_x
        h = max_y - min_y
        medium1 = (min_x + max_x) / 2
        medium2 = (min_y + max_y) / 2
        return medium1-w,medium2-h,medium1+w,medium2+h,
    else:
        return None,None,None,None
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
camera = cv2.VideoCapture(0)

result=None
while camera.isOpened():
    ret, frame = camera.read()
    x,y,x1,y1 = detector(frame, face, 1.2, 5)
    if (x is not None):
            dest_pts = np.float32([[x, y], [x1, y], [x1, y1], [x, y1]])
            src_pts = np.float32([[0, 0],
                                  [glass.shape[1], 0],
                                  [glass.shape[1], glass.shape[0]],
                                  [0, glass.shape[0]]
                                  ])

            M = cv2.getPerspectiveTransform(src_pts, dest_pts)
            persp_img = cv2.warpPerspective(glass, M, frame.shape[:2][::-1])

            gray = cv2.cvtColor(persp_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            fg = cv2.bitwise_and(persp_img, persp_img, mask=mask)

            result = cv2.add(bg, fg)
            if (result is not None):
                cv2.imshow("Camera", result)
            else:
                cv2.imshow("Camera", frame)
    else:
        if (result is not None):
            bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            fg = cv2.bitwise_and(persp_img, persp_img, mask=mask)
            result = cv2.add(bg, fg)
            cv2.imshow("Camera", result)
        else:
            cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()



