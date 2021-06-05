import cv2
import numpy as np
import utlis
from pythonRLSA import rlsa
import random

def drow_houghlines(img):


    imgCopy1=img.copy()
    imgCopy2=img.copy()
    #edges = cv2.Canny(img, 50, 150, apertureSize=3)
    kernel = np.ones((1,5 ))

    # lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    # lap = np.uint8(np.absolute(lap))

    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    sobelCombined = cv2.cvtColor(sobelCombined, cv2.COLOR_BGR2GRAY)

    ret3, imgThresholdBW = cv2.threshold(sobelCombined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    imgDialsobel = cv2.dilate(imgThresholdBW, kernel, iterations=1)  # APPLY DILATION
    imgThresholdsobel = cv2.erode(imgDialsobel, kernel, iterations=1)  # APPLY EROSION



    contours = cv2.findContours(imgThresholdsobel, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    #print("counters",contours)
    imgOriginal = cv2.cvtColor(imgCopy1, cv2.COLOR_BGR2GRAY)

    mood = utlis.getMostCommonPixel(imgOriginal)

    i = 0
    j = 0

    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        s = 0


        i = i + 1
        if h>10:cv2.rectangle(imgCopy1, (x+4, y+4), (x + w+2, y + h-4), (mood[0], mood[0], mood[0]), -1)




    img = cv2.cvtColor(imgCopy1, cv2.COLOR_RGB2GRAY)
    #ret3, imgThresholdBWInvert = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    ret3, bw_Invert = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel1 = np.ones((1, 3), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)



    img1 = cv2.erode(bw_Invert, kernel1, iterations=1)
    img2 = cv2.dilate(img1, kernel2, iterations=3)
    img3 = cv2.bitwise_and(bw_Invert, img2)
    img3 = cv2.bitwise_not(img3)
    img4 = cv2.bitwise_and(bw_Invert, bw_Invert, mask=img3)
    imgLines = cv2.HoughLinesP(img4, 15, np.pi / 180, 10, minLineLength=100, maxLineGap=15)

    #################################################################
    # kernelHorizontal = np.ones((1, 5))
    # kernelVerticle = np.ones((5, 1))
    # kernel = np.ones((3, 3))
    #
    #
    #
    # imgDial1 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernelHorizontal)
    # imgDial2 = cv2.morphologyEx(imgDial1, cv2.MORPH_OPEN, kernelVerticle)
    #
    # kernelHorizontal = np.ones((1, 5))
    # kernelVerticle = np.ones((5, 1))
    ######################################################################



    imgForHorizontalLine = cv2.dilate(imgDial2, kernelHorizontal, iterations=2)  # APPLY DILATION




    imgForVerticalLine = cv2.dilate(imgDial2, kernelVerticle, iterations=2)  # APPLY DILATION

    ret3, imgForHorizontalLine_Inver = cv2.threshold(imgForHorizontalLine, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret3, imgForVerticalLine_Invert = cv2.threshold(imgForVerticalLine, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    imgForHorizontalLine_Inver_Canny = cv2.Canny(imgForHorizontalLine_Inver, 50, 150, apertureSize=3)
    imgForVerticalLine_Invert_Canny = cv2.Canny(imgForVerticalLine_Invert, 50, 150, apertureSize=3)

    cv2.imshow("imgDialHori", imgForHorizontalLine_Inver_Canny)
    cv2.imshow("imgDialVeri", imgForVerticalLine_Invert_Canny)



    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    kernel1 = np.ones((1, 3), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)


    img1 = cv2.erode(imgForHorizontalLine, kernel1, iterations=1)
    img2 = cv2.dilate(img1, kernel2, iterations=3)
    img3 = cv2.bitwise_and(imgForHorizontalLine, img2)
    img3 = cv2.bitwise_not(img3)
    img4 = cv2.bitwise_and(imgForHorizontalLine, imgForHorizontalLine, mask=img3)
    imgLines = cv2.HoughLinesP(img4, 15, np.pi / 180, 10, minLineLength=100, maxLineGap=15)
    cv2.imshow("imgDialHori_New", img4)
    for i in range(len(imgLines)):
        for x1, y1, x2, y2 in imgLines[i]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    lines = cv2.HoughLinesP(imgForHorizontalLine_Inver_Canny, 1, np.pi/2, 100, minLineLength=100, maxLineGap=50)
    for line in lines:
         x1, y1, x2, y2 = line[0]
         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)



    lines = cv2.HoughLinesP(imgForVerticalLine_Invert_Canny, 1, np.pi / 2, 100, minLineLength=100, maxLineGap=50)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


    cv2.imshow('image11', img)

    # cv2.imshow('imgOriginal', imgOriginal)
    #
    # rotated90 = cv2.rotate(imgOriginal, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imshow('rotated90', rotated90)
    #
    # edges2 = cv2.Canny(rotated90, 50, 150, apertureSize=3)
    # cv2.imshow('edges2', edges2)
    #
    # lines2 = cv2.HoughLinesP(edges2, 1, np.pi, 100, minLineLength=100, maxLineGap=10)
    # #print("line2", lines2)
    # #lines = cv2.HoughLinesP(edges, 1, np.pi, 100, minLineLength=100, maxLineGap=10)
    # for line in lines2:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(rotated90, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow('image111', rotated90)
    return img

def drow_houghlines(img):


    imgCopy1=img.copy()
    imgCopy2=img.copy()
    #edges = cv2.Canny(img, 50, 150, apertureSize=3)
    kernel = np.ones((1,5 ))

    # lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    # lap = np.uint8(np.absolute(lap))

    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    sobelCombined = cv2.cvtColor(sobelCombined, cv2.COLOR_BGR2GRAY)

    ret3, imgThresholdBW = cv2.threshold(sobelCombined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    imgDialsobel = cv2.dilate(imgThresholdBW, kernel, iterations=1)  # APPLY DILATION
    imgThresholdsobel = cv2.erode(imgDialsobel, kernel, iterations=1)  # APPLY EROSION



    contours = cv2.findContours(imgThresholdsobel, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    #print("counters",contours)
    imgOriginal = cv2.cvtColor(imgCopy1, cv2.COLOR_BGR2GRAY)

    mood = utlis.getMostCommonPixel(imgOriginal)

    i = 0
    j = 0

    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        s = 0


        i = i + 1
        if h>10:cv2.rectangle(imgCopy1, (x+4, y+4), (x + w+2, y + h-4), (mood[0], mood[0], mood[0]), -1)




    img = cv2.cvtColor(imgCopy1, cv2.COLOR_RGB2GRAY)
    #ret3, imgThresholdBWInvert = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    ret3, bw_Invert = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel1 = np.ones((1, 3), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)



    img1 = cv2.erode(bw_Invert, kernel1, iterations=1)
    img2 = cv2.dilate(img1, kernel2, iterations=3)
    img3 = cv2.bitwise_and(bw_Invert, img2)
    img3 = cv2.bitwise_not(img3)
    img4 = cv2.bitwise_and(bw_Invert, bw_Invert, mask=img3)
    imgLines = cv2.HoughLinesP(img4, 15, np.pi / 180, 10, minLineLength=100, maxLineGap=15)

    #################################################################
    # kernelHorizontal = np.ones((1, 5))
    # kernelVerticle = np.ones((5, 1))
    # kernel = np.ones((3, 3))
    #
    #
    #
    # imgDial1 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernelHorizontal)
     imgDial2 = cv2.morphologyEx(imgDial1, cv2.MORPH_OPEN, kernelVerticle)
    #
    # kernelHorizontal = np.ones((1, 5))
    # kernelVerticle = np.ones((5, 1))
    ######################################################################



    imgForHorizontalLine = cv2.dilate(imgDial2, kernelHorizontal, iterations=2)  # APPLY DILATION




    imgForVerticalLine = cv2.dilate(imgDial2, kernelVerticle, iterations=2)  # APPLY DILATION

    ret3, imgForHorizontalLine_Inver = cv2.threshold(imgForHorizontalLine, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret3, imgForVerticalLine_Invert = cv2.threshold(imgForVerticalLine, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    imgForHorizontalLine_Inver_Canny = cv2.Canny(imgForHorizontalLine_Inver, 50, 150, apertureSize=3)
    imgForVerticalLine_Invert_Canny = cv2.Canny(imgForVerticalLine_Invert, 50, 150, apertureSize=3)

    cv2.imshow("imgDialHori", imgForHorizontalLine_Inver_Canny)
    cv2.imshow("imgDialVeri", imgForVerticalLine_Invert_Canny)



    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    kernel1 = np.ones((1, 3), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)


    img1 = cv2.erode(imgForHorizontalLine, kernel1, iterations=1)
    img2 = cv2.dilate(img1, kernel2, iterations=3)
    img3 = cv2.bitwise_and(imgForHorizontalLine, img2)
    img3 = cv2.bitwise_not(img3)
    img4 = cv2.bitwise_and(imgForHorizontalLine, imgForHorizontalLine, mask=img3)
    imgLines = cv2.HoughLinesP(img4, 15, np.pi / 180, 10, minLineLength=100, maxLineGap=15)
    cv2.imshow("imgDialHori_New", img4)
    for i in range(len(imgLines)):
        for x1, y1, x2, y2 in imgLines[i]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    lines = cv2.HoughLinesP(imgForHorizontalLine_Inver_Canny, 1, np.pi/2, 100, minLineLength=100, maxLineGap=50)
    for line in lines:
         x1, y1, x2, y2 = line[0]
         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)



    lines = cv2.HoughLinesP(imgForVerticalLine_Invert_Canny, 1, np.pi / 2, 100, minLineLength=100, maxLineGap=50)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


    cv2.imshow('image11', img)

    # cv2.imshow('imgOriginal', imgOriginal)
    #
    # rotated90 = cv2.rotate(imgOriginal, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imshow('rotated90', rotated90)
    #
    # edges2 = cv2.Canny(rotated90, 50, 150, apertureSize=3)
    # cv2.imshow('edges2', edges2)
    #
    # lines2 = cv2.HoughLinesP(edges2, 1, np.pi, 100, minLineLength=100, maxLineGap=10)
    # #print("line2", lines2)
    # #lines = cv2.HoughLinesP(edges, 1, np.pi, 100, minLineLength=100, maxLineGap=10)
    # for line in lines2:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(rotated90, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow('image111', rotated90)
    return img