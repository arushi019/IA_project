import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_detection(img,colorRangeMin,colorRangeMax):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    colorMask = cv2.inRange(hsv,colorRangeMin,colorRangeMax)
    binarize = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    binarize[colorMask!=0] = 255
    return binarize

def convert(img):
    yellowDetect = color_detection(img,np.array([15,100,120]),np.array([80,255,255]))
    yellowDetect = cv2.medianBlur(yellowDetect,3)

    whiteDetect = color_detection(img,np.array([0,0,0]),np.array([0,255,255]))
    whiteDetect = cv2.medianBlur(whiteDetect,3)
    # cv2.imshow("whiteDetection",whiteDetect)
    binary_img = np.zeros((img.shape[0],img.shape[1]))
    binary_img[(whiteDetect!=0) | (yellowDetect!=0)] = 255
    binary_img = binary_img.astype(np.uint8)
    binary_img = cv2.medianBlur(binary_img,5)

    cv2.imshow("Binary",binary_img)

    midPointx = img.shape[0]//2
    midPointy = img.shape[1]//2


    roi = binary_img[midPointx:,-300+midPointy:midPointy+300]
    # cv2.imshow("dd",roi)

    src = np.float32([[0, 187], [600, 187], [224, 30], [330, 30]])
    dst = np.float32([[150, 450], [350, 450], [150, 0], [350, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(roi, M, (450, 450))
    # cv2.imshow("ROI",warped_img)

    edges = cv2.Canny(roi[:,roi.shape[1]//2:],50,50,apertureSize = 3)
    cv2.imshow("Canny", edges)

    # edges = cv2.Canny(,50,150,apertureSize = 3)
    roi = cv2.cvtColor(roi,cv2.COLOR_GRAY2RGB)

    lines = cv2.HoughLines(edges,1,np.pi/180,2)
    for rho,theta in lines[0]:
        # print(theta)
        # if(theta>1 or theta <0.5):
        #     theta = 0.8
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 100*(-b))
        y1 = int(y0 + 100*(a))
        x2 = int(x0 - 100*(-b))
        y2 = int(y0 - 100*(a))
        print(theta)
        cv2.line(roi[:,roi.shape[1]//2:],(x1,y1),(x2,y2),(255,0,255),2)


    edges = cv2.Canny(roi[:,:roi.shape[1]//2],50,50,apertureSize = 3)
    cv2.imshow("Canny", edges)

    # edges = cv2.Canny(,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,1)
    try:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 100*(-b))
            y1 = int(y0 + 100*(a))
            x2 = int(x0 - 100*(-b))
            y2 = int(y0 - 100*(a))
            cv2.line(roi[:,:roi.shape[1]//2],(x1,y1),(x2,y2),(255,0,255),2)
        cv2.imshow('houghlines3',roi)
    except:
        print("ERROR: NO LANE DETECTED")

    # minLineLength = 1
    # maxLineGap = 1
    # lines = cv2.HoughLinesP(warped_img,1,np.pi/180,10,minLineLength,maxLineGap)
    # print(lines)
    # for x1,y1,x2,y2 in lines[0]:
    #     cv2.line(warped_img,(x1,y1),(x2,y2),(255,255,0),100)

    # cv2.imshow('ddd',warped_img)

    # midPointx = binary_img.shape[0]//2
    # midPointy = binary_img.shape[1]//2
    # roi = binary_img[midPointx:,-250+midPointy:midPointy+200]
    # src = np.float32([[50, 125], [407, 129], [0, 166], [450, 160]])
    # dst = np.float32([[150, 100], [350, 100], [150, 450], [350, 450]])
    # M = cv2.getPerspectiveTransform(src, dst)
    # warped_img = cv2.warpPerspective(roi, M, (450, 450))
    # plt.imshow(roi,cmap="gray")
    # plt.show()
    # cv2.imshow("bin",binary_img)
    # cv2.imshow("ee",roi)
