import sys
import cv2 as cv
import numpy as np
import math

def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, gray = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    edges = cv.Canny(gray, 200, 400, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50,maxLineGap=10)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(image,(x1,y1),(x2,y2),(238,130,238),5)
    cv.namedWindow('line_detect_possible_demo', cv.WINDOW_NORMAL)
    cv.imshow("line_detect_possible_demo",image)
    cv.waitKey(0)


def line_detection(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,200,400,apertureSize=3)    #apertureSize是sobel算子大小，只能为1,3,5，7
    lines = cv.HoughLines(edges,1,np.pi/180,200)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    print(len(lines))
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    cv.namedWindow('image line', cv.WINDOW_NORMAL)
    cv.imshow("image line",image)

def dilate_erode_combin(img, n1, n2):
    '''
    n1: number of dilate
    n2: number of erode
    '''
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_d = img.copy()
    for _ in range(n1):
        img_d = cv.dilate(img_d, kernel)
    for _ in range(n2):
        img = cv.erode(img, kernel)
    res = cv.absdiff(img_d, img)
    
    return res



src = cv.imread("C:/Users/Sonus/Desktop/python_test/order line.png")
img = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
imgCan = cv.Canny(img, 501, 255)
dst = cv.fastNlMeansDenoising(imgCan)
ret, th = cv.threshold(img, 127, 255, cv.THRESH_BINARY)


contours, hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
con = cv.drawContours(th, contours, -1, (255, 0, 0), 3)


res = dilate_erode_combin(src, 4, 1)
hsv = cv.cvtColor(res, cv.COLOR_BGR2HSV)
lower = np.array([50, 80, 80])
upper = np.array([70, 255, 255])
green_mask = cv.inRange(hsv, lower, upper)
res_green = cv.bitwise_and(res, res, mask=green_mask)
cv.namedWindow('res', cv.WINDOW_NORMAL)
cv.imshow('res', res_green)
cv.waitKey(0)






fld = cv.ximgproc.createFastLineDetector()
lines = fld.detect(th)

result_img = fld.drawSegments(img,lines)