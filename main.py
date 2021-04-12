import cv2
import numpy as np
import random

cap = cv2.VideoCapture("IMG_9803.MOV")
ret, frame = cap.read()
while(1):
    #print(ret, frame)
    ret, frame = cap.read()
    #print(type(frame))
    if str(type(frame)) == "<class 'NoneType'>":
        break
    output = frame.copy()
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    #nframe = cv2.convertScaleAbs(blur, alpha=2.0, beta=10)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    #gray[0:100, 0:1280] = np.ones((100,1280))
    #gray[300:720, 0:1280] = np.ones((420, 1280))
    cv2.imshow("",gray)
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        maxx = 0
        maxy = 0
        maxr = 0
        for (x, y, r) in circles:
            print(x, y, r)
            if r > maxr:
                maxx = x
                maxy = y
                maxr = r
        #cv2.circle(output, (maxx, maxy), maxr, (0, 165, 255), 4)
        cv2.circle(output, (maxx, maxy), maxr, (random.randrange(0,255), random.randrange(0,255), random.randrange(0,255)), 4)
        cv2.rectangle(output, (maxx - 5, maxy - 5), (maxx + 5, maxy + 5), (50, 50, 50), -1)

    cv2.imshow('frame',np.hstack([frame, output]))
    if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
       cap.release()
       cv2.destroyAllWindows()
       break
    #print(frame.shape)
    #print(output.shape)
    #print(gray.shape)
    #cv2.imshow('frame',np.hstack([frame, output, gray]))
