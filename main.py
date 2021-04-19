import cv2
import numpy as np
import random


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(image, pts, inve):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    if inve:
        M = np.linalg.pinv(M)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


cap = cv2.VideoCapture("IMG_0202.MOV")
ret, frame = cap.read()
framenum = 1
perc = 0
height, width, channels = frame.shape
print(frame.shape)
ptsstring = "[(-710,0),(0,{}),({},{}),({},0)]".format(720,1280,720,2000)
ptsstring = "[(0,0),(0,{}),({},{}),({},0)]".format(720,1280,720,1280)
pts = np.array(eval(ptsstring), dtype="float32")
while(1):
    #print(ret, frame)
    ret, frame = cap.read()
    #print(type(frame))
    if str(type(frame)) == "<class 'NoneType'>":
        break
    output = frame.copy()
    #ptsstring = "[(0,0),(0,{}),({},{}),({},0)]".format(720-((framenum/48)*(framenum-165)), 1280, 720, 1280)
    ptsstring = "[(0,0),(0,{}),({},{}),({},0)]".format(720 - (((framenum-75)**2 * (framenum-130)**2)/150000), 1280, 720, 1280)
    pts = np.array(eval(ptsstring), dtype="float32")
    output2 = four_point_transform(output, pts, False)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    #nframe = cv2.convertScaleAbs(blur, alpha=2.0, beta=10)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    image2 = four_point_transform(gray, pts, False)
    #gray[0:100, 0:1280] = np.ones((100,1280))
    #gray[300:720, 0:1280] = np.ones((420, 1280))
    cv2.imshow("",image2)
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(image2, cv2.HOUGH_GRADIENT, 1.2, 100)
    print("Frame: "+str(framenum))
    framenum+=1
    if circles is not None:
        perc+=1
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
        cv2.circle(output2, (maxx, maxy), maxr, (random.randrange(0,255), random.randrange(0,255), random.randrange(0,255)), 4)
        cv2.rectangle(output2, (maxx - 5, maxy - 5), (maxx + 5, maxy + 5), (0, 0, 0), -1)

    output3 = four_point_transform(output2, pts, True)
    #cv2.imshow("circle", output3)
    rows, cols, channels = output3.shape
    if rows > 720:
        rows = 720
    if cols >1280:
        cols = 1280
    if circles is not None:
        output[0:rows, 0:cols] = output3[0:rows, 0:cols]
    #cv2.imshow("output?",output)
    cv2.imshow('frame',np.hstack([frame, output]))

    #if circles is not None:
    #    cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
       cap.release()
       cv2.destroyAllWindows()
       break
    #print(frame.shape)
    #print(output.shape)
    #print(gray.shape)
    #cv2.imshow('frame',np.hstack([frame, output, gray]))
print(str(round(perc/(framenum-38-90)*100))+"%")
