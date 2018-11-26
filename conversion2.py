import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import sys

def drawLines(img, right_equation,left_equation,M):
    color_warp = np.zeros_like(img).astype(np.uint8)
    fitx = np.linspace(0, img.shape[1] - 1, img.shape[1])
    left_fity = left_equation(fitx)
    right_fity = right_equation(fitx)

    left_fity = np.array(left_fity,dtype=np.int32)
    fitx = np.array(fitx,dtype=np.int32)

    right_fity = np.array(right_fity,dtype=np.int32)

    rbg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in range(len(left_fity)):
        rbg = cv2.circle(rbg,(fitx[i],left_fity[i]),5,(0,255,0),-1)


    for i in range(len(right_fity)):
        rbg = cv2.circle(rbg,(fitx[i],right_fity[i]),5,(0,255,0),-1)
    # cv2.imshow("uiui",rbg)
    return rbg

def getSlope(x1,x2,y1,y2):
    return np.divide(y1-y2,x1-x2)
def findLanes(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("djnj",gray_image)


    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV), lower_yellow, upper_yellow)

    mask_white = cv2.inRange(gray_image, 200, 255)

    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    mask_yw_image = np.array(mask_yw_image,dtype="uint8")
    cv2.imshow("DD",mask_yw_image)


    medianBlur = cv2.medianBlur(mask_yw_image,5)
    gaussianBlur  = cv2.GaussianBlur(mask_yw_image,(5,5),0)
    # binary_img = cv2.medianBlur(mask_yw_image,3)
    cv2.imshow("mm",cv2.medianBlur(cv2.bilateralFilter(mask_yw_image,9,75,75),3))
    gaussianBlur = cv2.medianBlur(cv2.bilateralFilter(mask_yw_image,9,75,75),3)
    # return
    low_threshold = 50
    high_threshold = 150
    canny_edges = cv2.Canny(gaussianBlur,low_threshold,high_threshold)
    # cv2.imshow("DD",canny_edges)


    midPointx = img.shape[0]//2
    midPointy = img.shape[1]//2


    imshape = img.shape
    lower_left = [imshape[1]/5,imshape[0]]
    lower_right = [imshape[1]-imshape[1]/3.5,imshape[0]]
    top_left = [imshape[1]/2-imshape[1]/9,imshape[0]/2+imshape[0]/10]
    top_right = [imshape[1]/2+imshape[1]/30,imshape[0]/2+imshape[0]/10]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    temp_image = np.zeros_like(canny_edges)
    roi_image = cv2.fillPoly(temp_image, vertices,255)
    roi = cv2.bitwise_and(roi_image,canny_edges)
    # cv2.imshow("fff",roi)
    # return



    src = np.float32([lower_left, lower_right, top_left, top_right])
    dst = np.float32([[100, 450], [300, 450], [100, 0], [300, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(roi, M, (450,450))

    M_reverse = cv2.getPerspectiveTransform(dst,src)

    minLineLength = 200
    maxLineGap = 20
    sobel_x_img = np.uint8(cv2.Sobel(warped_img,cv2.CV_64F,1,0,ksize=5))
    # return
    cv2.imshow("sobelx",sobel_x_img)
    lines = cv2.HoughLinesP(sobel_x_img,1,np.pi/180,30,minLineLength,maxLineGap)
    # return
    left_lane_x = []
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []


    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if(getSlope(x1,x2,y1,y2)>10 and x1>midPointx and x2>midPointx ):# or getSlope(x1,x2,y1,y2)<-0.1):
                    cv2.line(warped_img,(x1,y1),(x2,y2),255,10)
                    right_lane_x.append(x1)
                    right_lane_x.append(x2)
                    right_lane_y.append(y1)
                    right_lane_y.append(y2)
                    print("right lane",getSlope(x1,x2,y1,y2))
                elif(getSlope(x1,x2,y1,y2)<-10 and x1<midPointx and x2<midPointx ):# or getSlope(x1,x2,y1,y2)<-0.1):
                    cv2.line(warped_img,(x1,y1),(x2,y2),255,10)
                    left_lane_x.append(x1)
                    left_lane_x.append(x2)
                    left_lane_y.append(y1)
                    left_lane_y.append(y2)                    # right_lane./append()
                    print("left lane",getSlope(x1,x2,y1,y2))
                # print(getSlope(x1,x2,y1,y2))

        cv2.imshow("lane",warped_img)

        left_lane_x = np.array(left_lane_x,dtype = np.float32)
        left_lane_y = np.array(left_lane_y,dtype = np.float32)
        right_lane_x = np.array(right_lane_x,dtype = np.float32)
        right_lane_y = np.array(right_lane_y,dtype = np.float32)

        # left_lane_x-=left_lane_x.mean()
        # right_lane_x-=right_lane_x.mean()
        # right_lane_y-=right_lane_y.mean()
        # left_lane_y-=left_lane_y.mean()

        try:
            right_parameters = np.polyfit(right_lane_x, right_lane_y, 1.8)
            right_equation = np.poly1d(right_parameters)

        except:
            right_equation = "-18.8 x + 5604"

        try:
            left_parameters = np.polyfit(left_lane_x, left_lane_y, 1.8)
            left_equation = np.poly1d(left_parameters)
        except:
            left_equation = "-13.63 x + 2069"

        print(right_equation,left_equation)
        detect = drawLines(warped_img,right_equation,left_equation,M)
        toBack = cv2.warpPerspective(detect, M_reverse, (img.shape[1],img.shape[0]))
        # cv2.imshow("BACK",toBack)
        total = cv2.addWeighted(img, 1, toBack,0.3, 0)
        cv2.imshow("total",total)
        # cv2.imwrite("correct.jpg",total)
        # cv2.imwrite("original.jpg",img)
    except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(e, exc_tb.tb_lineno)

    # for rho,theta in lines[0]:
    #     # print(theta)
    #     # if(theta>1 or theta <0.5):
    #     #     theta = 0.8
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 100*(-b))
    #     y1 = int(y0 + 100*(a))
    #     x2 = int(x0 - 100*(-b))
    #     y2 = int(y0 - 100*(a))
    #     print(theta)
    #     cv2.line(roi[:,roi.shape[1]//2:],(x1,y1),(x2,y2),(255,0,255),2)
    #
    #
    # edges = cv2.Canny(roi[:,:roi.shape[1]//2],50,50,apertureSize = 3)
    # cv2.imshow("Canny", edges)
    #
    # # edges = cv2.Canny(,50,150,apertureSize = 3)
    #
    # lines = cv2.HoughLines(edges,1,np.pi/180,1)
    # try:
    #     for rho,theta in lines[0]:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + 100*(-b))
    #         y1 = int(y0 + 100*(a))
    #         x2 = int(x0 - 100*(-b))
    #         y2 = int(y0 - 100*(a))
    #         cv2.line(roi[:,:roi.shape[1]//2],(x1,y1),(x2,y2),(255,0,255),2)
    #     cv2.imshow('houghlines3',roi)
    # except:
    #     print("ERROR: NO LANE DETECTED")


    #
    # roi2 = canny_edges[midPointx:,-300+midPointy:midPointy+300]
    # src = np.float32([[0, 187], [600, 187], [224, 30], [330, 30]])
    # dst = np.float32([[150, 450], [350, 450], [150, 0], [350, 0]])
    # M = cv2.getPerspectiveTransform(src, dst)
    # warped_img = cv2.warpPerspective(roi2, M, (450,450))
    # #
    # warped_img[:,:100] = 0
    # warped_img[:,375:] = 0
    #
    # M_reverse = cv2.getPerspectiveTransform(dst, src)
    # reverse_warp = cv2.warpPerspective(warped_img, M_reverse, (gray_image.shape[1],gray_image.shape[0]))
    # cv2.imshow("dd",reverse_warp)

    # yellowDetect = color_detection(img,np.array([15,100,120]),np.array([80,255,255]))
    # whiteDetect = cv2.inRange(img,200,255)
    #
    #
    # binary_img = np.zeros((img.shape[0],img.shape[1]))
    # binary_img[(whiteDetect!=0) | (yellowDetect!=0)] = 255
    #
    # cv2.imshow("DD",whiteDetect)
