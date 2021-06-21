import cv2
import numpy as np
import utlis
from pythonRLSA import rlsa
import random
import math


heightImg = 640
widthImg = 480
utlis.initializeTrackbars()


def getXFromRectx(item):
    return item[0]


def getXFromRecty(item):
    return item[1]


def getMostCommonPixel(image):
   # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #image1 = Image.fromarray(image, 'RGB')
    histogram = {}  #Dictionary keeps count of different kinds of pixels in image


    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            pixel = image.item(j, i)
            if pixel in histogram:
                histogram[pixel] += 1  # Increment count
            else:
                histogram[pixel] = 1  # pixel_val encountered for the first time

def remove_text(img,imgOrginal,imgConters):
    img2=cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2RGB)
    img3=cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2RGB)

    #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #sobelX = cv2.Sobel(img, cv2.CV_8U, 1, 0)
    #sobelY = cv2.Sobel(img, cv2.CV_8U, 0, 1)
    #sobelCombined = cv2.bitwise_and(sobelX, sobelY)
    kernel = np.ones((2,2))
    gradiant = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    rect = cv2.morphologyEx(gradiant, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((2, 2))
    dialate = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)


    #imgAdaptiveThre = cv2.adaptiveThreshold(gradiant, 255, 1, 1, 7, 2)
    ret3, otsu = cv2.threshold(gradiant, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blankImage1 = cv2.resize(otsu, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("otsu1", blankImage1)
    for cntr in imgConters:
        x, y, w, h = cv2.boundingRect(cntr)
        if (((x == 0) & (y == 0)) | ((x == 0) & (y + h == otsu.shape[0])) | ((x + w == otsu.shape[1]) & (y == 0)) | (
                ((x + w) == otsu.shape[1]) & ((y + h) == otsu.shape[0]))) | (
                h * w < imgOrginal.shape[0] * imgOrginal.shape[1] * 0.0009) | (
                (w / h > 8 and w > 50) | (h / w > 8 and h > 50))|(h>otsu.shape[0]/2):
            continue

        otsu=cv2.rectangle(otsu, (x-8 if x>=8 else 0, y-8 if y>=8 else 0), (x+w+10 if x+w+10<=otsu.shape[1] else otsu.shape[1], y + h+10  if y + h+10<=otsu.shape[0] else otsu.shape[0]), (0, 0,0), -1)
    ret3, otsu = cv2.threshold(otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = np.ones((1, 10))
    # otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((10, 1))
    # otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)


    #bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    # cv2.imshow("bw",bw)
    otsuCanny=cv2.Canny(otsu, 50, 150, apertureSize=3)
    cv2.imwrite("gradiant.jpg", otsu)
    blankImage1 = cv2.resize(otsu, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("otsu", blankImage1)
    # blankImage2 = cv2.resize(otsuCanny, (widthImg, heightImg))  # RESIZE IMAGE
    # cv2.imshow("canny", blankImage2)
    contours = cv2.findContours(otsuCanny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    rects = []
    blankImage = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    blankImage2 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    line_contours=[]
    for cntr in contours:

        x, y, w, h = cv2.boundingRect(cntr)
        if(((w/h)>8) & (w>50)):
            line_contours.append(cv2.boundingRect(cntr))
            img2=cv2.line(img2, (x, y), (x+w, y ), ( 255,0, 0), 5)
            img2=cv2.line(img2, (x, y+h), (x+w, y+h ),  ( 255,0, 0), 5)
            blankImage=cv2.line(blankImage, (x, y), (x+w, y ), (255, 255, 255), 5)
            blankImage=cv2.line(blankImage, (x, y+h), (x+w, y+h ), (255, 255, 255), 5)


        elif (((h/w)>8) & (h>50)):
            line_contours.append(cv2.boundingRect(cntr))

            img2 = cv2.line(img2, (x, y), (x , y+h), (0, 255, 0), 5)
            img2 = cv2.line(img2, (x+w, y ), (x + w, y + h), (0, 255, 0), 5)
            # blankImage = cv2.line(blankImage, (x, y), (x , y+h), (255, 255, 255), 5)
            # blankImage = cv2.line(blankImage, (x+w, y ), (x + w, y + h), (255, 255, 255), 5)
            blankImage2=cv2.line(blankImage2, (x, y), (x, y+h ), (255, 255, 255), 10)


        else:
            img2 = cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
    blankImage = cv2.resize(blankImage2, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("textRMV_R44444", blankImage)

    img_for_line_length = blankImage2.copy()
    kernel = np.ones((2, 20))
    img_for_line_lengthDial = cv2.dilate(img_for_line_length, kernel, iterations=1)  # APPLY DILATION
    img_for_line_lengthErod = cv2.erode(img_for_line_lengthDial, kernel, iterations=1)  # APPLY DILATION
    # img_for_line_lengthErodgray = cv2.cvtColor(img_for_line_lengthErod, cv2.COLOR_GRAY2RGB)
    ret3, img_for_line_lengthErodgray = cv2.threshold(cv2.cvtColor(img_for_line_lengthErod, cv2.COLOR_RGB2GRAY), 0, 255,
                                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours_linesForSumOfLen = cv2.findContours(img_for_line_lengthErodgray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours_linesForSumOfLen = contours_linesForSumOfLen[0] if len(contours_linesForSumOfLen) == 2 else \
    contours_linesForSumOfLen[1]
    length_of_real_lines=0
    for cntr in contours_linesForSumOfLen:
        x, y, w, h = cv2.boundingRect(cntr)
        length_of_real_lines = length_of_real_lines + h

#     # kernel = np.ones((5, 5))
#     # dilate = cv2.dilate(blankImage, kernel, iterations=3)  # APPLY DILATION
#     # dilate = cv2.cvtColor(dilate,cv2.COLOR_RGB2GRAY)
#     # otsuCanny = cv2.Canny(dilate, 50, 150, apertureSize=3)
#     # lines_H = cv2.HoughLinesP(otsuCanny, 1, np.pi / 360, 200, minLineLength=int(img.shape[0] * .008), maxLineGap=int(img.shape[0] * .05))
#     #
#     #
# a
    return img2,line_contours,length_of_real_lines,contours_linesForSumOfLen


def drowLines(img,imgOriginal,thresh,rmv):# rmv=remove_images(imgCopy1,imgCopy2,thresh)
    imgCopy1 = img.copy()
    imgCopy2 = img.copy()
    imgCopy3 =  np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    imgCopy4 =  np.zeros((img.shape[0], img.shape[1], 3), np.uint8)


    img_withowt_pics=rmv[3]
    img_Conters=rmv[4]
    textRMV=remove_text(rmv[0], imgOriginal,rmv[4])
    contours=textRMV[1]


    kernel = np.ones((1, 5))

    sobelX = cv2.Sobel(img_withowt_pics, cv2.CV_16U, 1, 0)
    sobelY = cv2.Sobel(img_withowt_pics, cv2.CV_16U, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))


    for cntr in contours:

        x, y, w, h = cntr
        if(((w/h)>8) & (w>50)):

            #sobelY=cv2.line(sobelY, (x, y), (x+w, y ),(255, 255, 255) , 5)

            sobelX=cv2.line(sobelX, (x, y), (x+w, y ),(0, 0, 0) , 5)

            imgCopy3 = cv2.line(imgCopy3, (x, y), (x + w, y), (255, 255, 255), 5)
            imgCopy3 = cv2.line(imgCopy3, (x, y + h), (x + w, y + h), (255, 255, 255), 5)

        elif (((h/w)>8) & (h>50)):
            # sobelY=cv2.line(sobelY, (x, y), (x+w, y ),(0, 0, 0), 5)
            sobelY=cv2.line(sobelY, (int(x+w/2), y ), (int(x+w/2), y + h), (0, 0, 0), 10)

            imgCopy4=cv2.line(imgCopy4, (int(x+w/2), y), (int(x+w/2), y + h),(0, 0, 255), 10)

            # sobelX=cv2.line(sobelX, (x, y), (x+w, y ),(255, 255, 255), 5)
            sobelX=cv2.line(sobelX, (int(x+w/2), y ), (int(x+w/2), y + h), (255, 255, 255), 5)

    # kernel = np.ones((5,50))
    #
    # imgCopy3 = cv2.dilate(imgCopy3, kernel, iterations=2)  # APPLY DILATION
    # imgCopy3 = cv2.erode(imgCopy3, kernel, iterations=2)  # APPLY DILATION
    # imgCopy3=cv2.cvtColor(imgCopy3,cv2.COLOR_RGB2GRAY)
    ############################################################################


    kernel = np.ones((50, 2))
    imgCopy4 = cv2.dilate(imgCopy4, kernel, iterations=2)  # APPLY DILATION
    imgCopy4 = cv2.erode(imgCopy4, kernel, iterations=2)  # APPLY DILATION
    kernel = np.ones((10, 10))


    imgCopy4 = cv2.dilate(imgCopy4, kernel, iterations=1)  # APPLY DILATION
    imgCopy4 = cv2.erode(imgCopy4, kernel, iterations=1)  # APPLY DILATION


    ret3, otsu_hori_lines = cv2.threshold(cv2.cvtColor(imgCopy4, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours_hori_lines = cv2.findContours(otsu_hori_lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    contours_hori_lines = contours_hori_lines[0] if len(contours_hori_lines) == 2 else contours_hori_lines[1]
    # lines = cv2.HoughLinesP(imgCopy3, 5, np.pi/2, 100, minLineLength=int(imgCopy3.shape[0] / 7),maxLineGap=int(imgCopy3.shape[0] * .009))
    y_lines = []
    for line in contours_hori_lines:
        x1, y1, x2, y2 = cv2.boundingRect(line)
        cv2.line(imgCopy2, (int(x1 + (x2 / 2)), y1), (int(x1 + (x2 / 2)), y1 + y2), (255, 255, 255),10)
        y_lines.append([x1, y1, x2, y2, False])
    ####################################################################################

    kernel = np.ones((2, 50))
    imgCopy3 = cv2.dilate(imgCopy3, kernel, iterations=2)  # APPLY DILATION
    imgCopy3 = cv2.erode(imgCopy3, kernel, iterations=3)  # APPLY DILATION
    kernel = np.ones((10, 10))
    imgCopy3 = cv2.dilate(imgCopy3, kernel, iterations=1)  # APPLY DILATION
    imgCopy3 = cv2.erode(imgCopy3, kernel, iterations=1)  # APPLY DILATION
    ret3, otsu = cv2.threshold(cv2.cvtColor(imgCopy3, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



    contours_l = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    contours_l = contours_l[0] if len(contours_l) == 2 else contours_l[1]
    # lines = cv2.HoughLinesP(imgCopy3, 5, np.pi/2, 100, minLineLength=int(imgCopy3.shape[0] / 7),maxLineGap=int(imgCopy3.shape[0] * .009))
    x_line = []
    for line in contours_l:
        x1, y1, x2, y2 =cv2.boundingRect(line)
        cv2.line(imgCopy2, (x1,int( y1+(y2/2))), (x1+x2, int(y1+(y2/2))), (255, 255, 0), int(heightImg / 350))
        x_line.append([ x1, y1, x2, y2, False])
    ####################################################################################
    imgAllLins = cv2.resize(imgCopy2, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("imgAllLins_c", imgAllLins)



    x_line.sort(key=getXFromRecty)
    i = 0
    j = 0
    maxWidth = 0
    maxWidthIndex = 0
    for line in x_line:
        x1, y1, w1, h1, s1 = line

        if imgCopy3.shape[0] / 4 > y1 + h1 / 2:
            if (maxWidth < w1):
                maxWidth = w1
                maxWidthIndex = int(y1 + h1 / 2)
                j = i
        else:
               break
    i = i + 1
    x_lineCopy = []
    upperBoder = 0


    if (maxWidth > (imgCopy3.shape[1] *3)/ 5):
        x_line = []
        k = 0
        upperBoder = maxWidthIndex

    #####################################################################################


    kernel = np.ones((1, 3))
    erodeY = cv2.erode(sobelY, kernel, iterations=1)  # APPLY DILATION
    imgDial = cv2.dilate(erodeY, kernel, iterations=1)  # APPLY DILATION

    kernelX = np.ones((3, 1))
    erodeX = cv2.erode(sobelX, kernelX, iterations=1)  # APPLY DILATION
    imgDialX = cv2.dilate(erodeX, kernelX, iterations=1)  # APPLY DILATION




    ret3, imgDial_I = cv2.threshold(imgDial, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # imgDial_I = cv2.erode(imgDial_I, kernel, iterations=1)  # APPLY DILATION
    # imgDial_I = cv2.dilate(imgDial_I, kernel, iterations=1)  # APPLY DILATION
    length_of_real_lines=textRMV[2]
    kernel = np.ones((15, 15))
    # if ((length_of_real_lines ) > img.shape[0] * 1.5):
    #     kernel = np.ones((10, 10))

    imgDial_I = cv2.dilate(imgDial_I, kernel, iterations=1)  # APPLY DILATION
    # imgDial_I = cv2.dilate(imgDial_I, kernel, iterations=1)  # APPLY DILATION
    blankImageX = cv2.resize(imgCopy3, (widthImg, heightImg))  # RESIZE IMAGE

    cv2.imshow("lines", blankImageX)
    ret3, imgDial_I = cv2.threshold(imgDial_I, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    image_rlsa_hori = rlsa.rlsa(image=imgDial_I, horizontal=False, vertical=True, value=imgDial_I.shape[0]/40)

    ret3, imgDial_IX = cv2.threshold(imgDialX, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    image_rlsa_X = rlsa.rlsa(image=imgDial_IX, horizontal=True , vertical=False, value=imgDial_I.shape[1]/40)


    #############

    for cntr in img_Conters:
        x, y, w, h = cv2.boundingRect(cntr)
        # epsilon = 0.01 * cv2.arcLength(cntr, True)
        # approx = cv2.approxPolyDP(cntr, epsilon, True)
        # hull = cv2.convexHull(cntr)

        if (((x == 0) & (y == 0)) | ((x == 0) & (y + h == image_rlsa_hori.shape[0])) | ((x+w == image_rlsa_hori.shape[1]) &(y == 0)) | (
                ((x+w) == image_rlsa_hori.shape[1]) & ((y+h) == image_rlsa_hori.shape[0]))) | (h * w < imgOriginal.shape[0] * imgOriginal.shape[1] * 0.0009):

            continue
        # img_with_mood_boxes=cv2.rectangle(img_with_mood_boxes, (x, y), (x + w, y + h), (mood[0], mood[0], mood[0]), -1)
        # imgDial_v=cv2.rectangle(imgDial_v, (x, y), (x + w, y + h), (0,0, 0), -1)
        image_rlsa_hori=cv2.rectangle(image_rlsa_hori, (x, y), (x + w, y + h), (0,0,0), -1)
        image_rlsa_X=cv2.rectangle(image_rlsa_X, (x, y), (x + w, y + h), (0,0,0), -1)
        # img_with_wight_boxes=cv2.rectangle(img_with_wight_boxes, (x, y), (x + w, y + h), (255, 255, 255), -1)

    #############



    # imgDial_IR = cv2.resize(imgDial, (widthImg, heightImg))  # RESIZE IMAGE
    # cv2.imshow("imgDial_IR", imgDial_IR)

    kernel = np.ones((5, 5))

    image_rlsa_hori_dilate = cv2.erode(image_rlsa_hori, kernel, iterations=1)  # APPLY DILATION
    kernelx = np.ones((15, 2))
    image_rlsa_x_dilate = cv2.erode(image_rlsa_X, kernelx, iterations=1)  # APPLY DILATION
    for cntr in contours:

        x, y, w, h = cntr
        if(((w/h)>8) & (w>50)):

            #sobelY=cv2.line(sobelY, (x, y), (x+w, y ),(255, 255, 255) , 5)

            sobelX=cv2.line(sobelX, (x, y), (x+w, y ),(255, 255, 255) , 5)

        elif (((h/w)>8) & (h>img.shape[0]/50)):
            # image_rlsa_hori_dilate=cv2.line(image_rlsa_hori_dilate, (x, y), (x+w, y ),(255, 255, 255), 5)
            image_rlsa_hori_dilate=cv2.line(image_rlsa_hori_dilate, (int(x+w/2), y ), (int(x+w/2), y + h), (255, 255, 255), 10)

            # sobelX=cv2.line(sobelX, (x, y), (x+w, y ),(0, 0, 0), 5)
            sobelX=cv2.line(sobelX, (int(x+w/2), y ), (int(x+w/2), y + h), (0, 0, 0), 10)
    image_rlsa_hori_dilate = cv2.rectangle(img=image_rlsa_hori_dilate, pt1=(0, 0),
                               pt2=(image_rlsa_hori_dilate.shape[1], upperBoder), color=(0, 0, 0),
                               thickness=-1)
    blankImageX = cv2.resize(image_rlsa_hori_dilate, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("textRMV_Rsla1", blankImageX)
    for line in y_lines:
        x1, y1, w1, h1, s1 = line
        image_rlsa_hori_dilate = cv2.line(image_rlsa_hori_dilate, (int(x1 + w1 / 2), y1), (int(x1 + w1 / 2), y1 + h1), (255, 255, 255), 10)

    lines_V = cv2.HoughLinesP(image_rlsa_hori_dilate, 5, np.pi, 100, minLineLength=int(image_rlsa_hori_dilate.shape[0] /7), maxLineGap=int(imgOriginal.shape[0] * .009))
    blankImage = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)



    if (lines_V is not None):
        for line in lines_V:
            x1, y1, x2, y2 = line[0]
            cv2.line(blankImage, (x1, y1), (x2, y2), (255, 255, 255), int(heightImg / 350))
    blankImageX = cv2.resize(blankImage, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("textRMV_Rsla", blankImageX)
    blankImage = cv2.rectangle(img=blankImage, pt1=(0, 0),
                                            pt2=(blankImage.shape[1], upperBoder), color=(0, 0, 0),
                                            thickness=-1)
    # blankImage = cv2.cvtColor(blankImage, cv2.COLOR_BGR2GRAY)
    # textRMV_R = cv2.resize(blankImage, (widthImg, heightImg))  # RESIZE IMAGE


    kernel = np.ones((5, 5))
    blankImage = cv2.dilate(blankImage, kernel, iterations=2)  # APPLY EROSION

    blankImage = cv2.resize(blankImage, (widthImg, heightImg))  # RESIZE IMAGE

    blankImageX = cv2.resize(image_rlsa_x_dilate, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("textRMV_R", blankImage)
    ret3, blankImage = cv2.threshold(cv2.cvtColor(blankImage,cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret3, blankImageX = cv2.threshold(blankImageX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blankimCopy=blankImageX.copy()
    blankImage[blankImage == 0] = 0
    blankImage[blankImage == 255] = 1
    vertical_projection = np.sum(blankImage, axis=0) * widthImg / heightImg

    blankImageX[blankImageX == 0] = 0
    blankImageX[blankImageX == 255] = 1
    x_projection = np.sum(blankImageX, axis=1) * widthImg / heightImg


    blankImageforVerticle = np.zeros((heightImg, widthImg, 3), np.uint8)
    blankImageforX = np.zeros((heightImg, widthImg, 3), np.uint8)

    for col in range(0, widthImg):
        cv2.line(blankImageforVerticle, (col, heightImg), (col, heightImg - int(vertical_projection[col])),
                 (255, 255, 255), 1)
    for row in range(heightImg):
        cv2.line(blankImageforX, (0, row),
                 (int(x_projection[row]), row),
                 (255, 255, 255), 1)
    # cv2.imshow("blankImageforVerticle",blankImageforVerticle)
    # cv2.imshow("test",blankImageforVerticle)



    v_projection_C=blankImageforVerticle.copy()
    x_projection_C=blankImageforX.copy()
    cv2.imshow("hori",x_projection_C)
    blankImageforHorizontal=cv2.rectangle(img=x_projection_C, pt1=(0, heightImg), pt2=(int((widthImg * widthImg * 0.98) / heightImg), 0), color=(0, 0, 0), thickness=-1)
    blankImageforHorizontal = cv2.cvtColor(blankImageforHorizontal, cv2.COLOR_BGR2GRAY)
    #kernel = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.uint8)
    kernel = np.ones((3, 2))
    blankImageforHorizontal = cv2.dilate(blankImageforHorizontal, kernel, iterations=2)  # APPLY DILATION
    blankImageforHorizontal = cv2.erode(blankImageforHorizontal, kernel, iterations=2)  # APPLY DILATION



    contours_h = cv2.findContours(blankImageforHorizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    contours_h = contours_h[0] if len(contours_h) == 2 else contours_h[1]
    rects_h = []

    for cntr in contours_h:
        x, y, w, h = cv2.boundingRect(cntr)
        x, y, w, h = int(x * img.shape[1] / widthImg), int(y * img.shape[0] / heightImg), int(
            w * img.shape[1] / widthImg),int(h * img.shape[0] / heightImg)
        rects_h.append([x, y, w, h])
    rects_h.sort(key=getXFromRecty)

    upperConer = 0, 0, 0, 0
    bottomConer = [0, img.shape[0], 0, 0]
    if len(rects_h) > 1:
        if (rects_h[0][1] < 100):
            upperConer = [0, rects_h[0][1],0, rects_h[0][3]  ]
        if (img.shape[0] - rects_h[len(rects_h) - 1][1] - rects_h[len(rects_h) - 1][3] <img.shape[0]/ 100):
            # rightConer =rects_v[len(rects_v)-1] **imgOrginal.imgThresholdBW.shape[1]/widthImg
            bottomConer =[0, rects_h[len(rects_h) - 1][1],0, rects_h[len(rects_h) - 1][3]  ]

    kernel = np.ones((2, 2))
    blankImageforVerticle = cv2.dilate(blankImageforVerticle, kernel, iterations=1)  # APPLY EROSION
    _,blankImageforVerticle = cv2.threshold(cv2.cvtColor(blankImageforVerticle,cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



    mean_vertical = int(np.mean(vertical_projection))
    max_vertical = int(np.max(vertical_projection))
    blankImageforVerticle_2=blankImageforVerticle.copy()
    cv2.rectangle(img=blankImageforVerticle, pt1=(0, heightImg),
                  pt2=(widthImg, int((mean_vertical*3))), color=(0, 0, 0), thickness=-1)
    contours_v = cv2.findContours(blankImageforVerticle, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours_v = contours_v[0] if len(contours_v) == 2 else contours_v[1]
    maxWidth = 0


    rects = []
    for cntr in contours_v:
        rects.append(cv2.boundingRect(cntr))
        x, y, w, h = cv2.boundingRect(cntr)


        if( (w > maxWidth) and (x != 0) and ((x + w) != widthImg)) |((w > maxWidth) and (w > widthImg/5)):
            maxWidth = w

    ###############################################################
    ##############################################################
    kernel = np.ones((10, 10))
    blankImageforVerticle_2 = cv2.dilate(blankImageforVerticle_2, kernel, iterations=1)  # APPLY EROSION
    blankImageforVerticle_2 = cv2.erode(blankImageforVerticle_2, kernel, iterations=1)  # APPLY EROSION
    # img2=cv2.rectangle(blankImageforVerticle, (x, y), (x + w, y + h), (255, 0, 0), 1)

    bw_for_bottome_box = blankImageforVerticle_2.copy()
    bw_for_bottome_box[bw_for_bottome_box == 0] = 0
    bw_for_bottome_box[bw_for_bottome_box == 255] = 1
    x_projection_for_bottom_box = np.sum(bw_for_bottome_box, axis=1) * widthImg / heightImg
    blankImageforX_for_bottom_box  = np.zeros((heightImg, widthImg, 3), np.uint8)

    for row in range(heightImg):
        cv2.line(blankImageforX_for_bottom_box, (0, row),
                 (int(x_projection_for_bottom_box[row]), row),
                 (255, 255, 255), 1)

    blankImageforX_for_bottom_box= cv2.cvtColor(blankImageforX_for_bottom_box,cv2.COLOR_RGB2GRAY)
    contour = cv2.findContours(blankImageforX_for_bottom_box, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contour = contour[0] if len(contour) == 2 else contour[1]
    x, y, w, h = cv2.boundingRect(contour[0])
    blankImageforX_for_bottom_box=cv2.rectangle(img=blankImageforX_for_bottom_box, pt1=(0, heightImg), pt2=(int(w  * 0.98), 0), color=(0, 0, 0), thickness=-1)

    cv2.imshow("test", blankImageforVerticle_2)
    cv2.waitKey(100)
    contour= cv2.findContours(blankImageforX_for_bottom_box, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contour = contour[0] if len(contour) == 2 else contour[1]#############################################################################################

    if(len(contour)>0):
        x, y, w, h = cv2.boundingRect(contour[0])
        if(x>(widthImg*widthImg*3)/(heightImg*10)):

            cv2.rectangle(img=v_projection_C, pt1=(0, heightImg),
                          pt2=(widthImg, int(heightImg-(h*5/4))), color=(0, 0, 0), thickness=-1)
            cv2.rectangle(img=blankImageforVerticle_2, pt1=(0, heightImg),
                                  pt2=(widthImg, int(heightImg-(h*5/4))), color=(0, 0, 0), thickness=-1)



    cv2.imshow("test2", blankImageforVerticle_2)
    kernel = np.ones((15, 15))
    v_projection_C = cv2.dilate(v_projection_C, kernel, iterations=1)  # APPLY EROSION
    v_projection_C = cv2.erode(v_projection_C, kernel, iterations=1)  # APPLY EROSION
    _, v_projection_C = cv2.threshold(cv2.cvtColor(v_projection_C, cv2.COLOR_RGB2GRAY), 0, 255,
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # v_projection_C = cv2.cvtColor(v_projection_C, cv2.COLOR_RGB2GRAY)
    contours_v = cv2.findContours(v_projection_C, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours_v = contours_v[0] if len(contours_v) == 2 else contours_v[1]

    cv2.imshow("ttttttt", v_projection_C)

    for cntr in contours_v:
            x, y, w, h = cv2.boundingRect(cntr)

            v_projection_C=cv2.rectangle(v_projection_C, (x-(5 if x>5 else 0), y+20), (x + w+(5 if x+w+5<v_projection_C.shape[1] else 0),  v_projection_C.shape[0]), (0,0,0), -1)
            # img_with_wight_boxes=cv2.rectangle(img_with_wight_boxes, (x, y), (x + w, y + h), (255, 255, 255), -1)

    kernel = np.ones((5, 5))
    v_projection_C = cv2.dilate(v_projection_C, kernel, iterations=1)  # APPLY EROSION
    v_projection_C = cv2.erode(v_projection_C, kernel, iterations=1)  # APPLY EROSION
    # v_projection_C=cv2.cvtColor(v_projection_C,cv2.COLOR_RGB2GRAY)
    contours_v = cv2.findContours(v_projection_C, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours_v = contours_v[0] if len(contours_v) == 2 else contours_v[1]
    rects = []
    for cntr in contours_v:
        x,y,w,h=cv2.boundingRect(cntr)
        x,y,w,h=int(x * img.shape[1] / widthImg), int(y * img.shape[0] / heightImg), int(w * img.shape[1] / widthImg), 0
        rects.append([x,y,w,h])
    rects.sort(key=getXFromRectx)
    leftConer = 0, 0, 0, 0
    rightConer = [int(img.shape[1]), 0, 0, 0]
    if len(rects) > 1:
        if (rects[0][0] < img.shape[1]/15):
            leftConer = [rects[0][0] , rects[0][1],
                         rects[0][2] , 0]
        if (widthImg - rects[len(rects) - 1][0] - rects[len(rects) - 1][2] < img.shape[1]/15):
            # rightConer =rects_v[len(rects_v)-1] **imgOrginal.imgThresholdBW.shape[1]/widthImg
            rightConer = [rects[len(rects) - 1][0] , rects[len(rects) - 1][1],
                          rects[len(rects) - 1][2] , 0]


    blankImage = cv2.resize(v_projection_C, (widthImg, heightImg))  # RESIZE IMAGE

    cv2.imshow("blankImage_11", blankImage)
    backgraond_line = lines_V
    num_of_effective_col = len(rects)
    if (leftConer[2] > 0):
        num_of_effective_col = num_of_effective_col - 1
    if (rightConer[2] > 0):
        num_of_effective_col = num_of_effective_col - 1

    return [leftConer,rightConer],[upperConer,bottomConer],maxWidth,rects,rects_h,backgraond_line,num_of_effective_col #rects=collems rects_h=vertical separethins


def remove_images(img,imgOrginal,tresh):

    kernalSize=int( img.shape[1] / widthImg/20)
    if ( kernalSize%2==0):kernalSize=kernalSize+1
    CropedImg1 = cv2.GaussianBlur(img,(kernalSize , kernalSize), 0)  # gaussian
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel = np.ones((5, 5))#10
    # grayC=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
    imgDial = cv2.dilate(CropedImg1, kernel, iterations=4)  # APPLY DILATION
    #cv2.imshow("dilate", imgDial)
    kernel = np.ones((5, 5))
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION


    imgThreshold = cv2.GaussianBlur(imgDial,(kernalSize , kernalSize), 0)  # gaussian

    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(imgThreshold, -1, kernel)  # applying the sharpening kernel to the input image & displaying it.
    # sharpened = cv2.Canny(sharpened, 50, 150, apertureSize=3)

    ret3, bw1 = cv2.threshold(sharpened, tresh , 255, cv2.THRESH_BINARY_INV)
    bw1 = cv2.dilate(bw1, kernel, iterations=2)  # APPLY EROSION

    contours = cv2.findContours(bw1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    sImg = cv2.resize(imgOrginal, (int(imgThreshold.shape[1] * widthImg / imgOrginal.shape[1]),
                                        int(imgThreshold.shape[0] * heightImg / imgOrginal.shape[0])))

    mood = utlis.getMostCommonPixel(sImg)
    img_with_wight_boxes = img.copy()
    img_with_mood_boxes = img.copy()
    img_with_black_boxes = img.copy()
    img_withowt_image = img.copy()
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        epsilon = 0.01 * cv2.arcLength(cntr, True)
        approx = cv2.approxPolyDP(cntr, epsilon, True)
        hull = cv2.convexHull(cntr)

        if (((x == 0) & (y == 0)) | ((x == 0) & (y + h == img.shape[0])) | ((x+w == img.shape[1]) &(y == 0)) | (
                ((x+w) == img.shape[1]) & ((y+h) == img.shape[0]))) | (h * w < imgOrginal.shape[0] * imgOrginal.shape[1] * 0.0009)|((w/h>8 and w>50) | (h/w>8 and h>50) ):

            continue
        #img_with_mood_boxes=cv2.rectangle(img_with_mood_boxes, (x, y), (x + w, y + h), (mood[0], mood[0], mood[0]), -1)
        img_with_mood_boxes=cv2.rectangle(img_with_mood_boxes, (x-8 if x>=8 else 0, y-8 if y>=8 else 0), (x+w+8 if x+w+8<=img_with_mood_boxes.shape[1] else img_with_mood_boxes.shape[1], y + h+8  if y + h+8<=img_with_mood_boxes.shape[0] else img_with_mood_boxes.shape[0]), (mood[0], mood[0], mood[0]), -1)

        img_withowt_image = cv2.drawContours(img_withowt_image, [hull], -1, (mood[0], mood[0], mood[0]), thickness=-1)
        img_withowt_image = cv2.drawContours(img_withowt_image, [hull], -1, (mood[0], mood[0], mood[0]), thickness=10)

        img_with_black_boxes=cv2.rectangle(img_with_black_boxes, (x, y), (x + w, y + h), (0,0, 0), -1)
        img_with_wight_boxes=cv2.rectangle(img_with_wight_boxes, (x, y), (x + w, y + h), (255, 255, 255), -1)
        #
        # img_withowt_image=cv2.drawContours(img_withowt_image, [cntr], -1, (mood[0], mood[0], mood[0]), -1)
        # img_withowt_image=cv2.drawContours(img_withowt_image, [cntr], -1, (mood[0], mood[0], mood[0]), 10)

    cropedImg2 = cv2.resize(img_with_mood_boxes, (widthImg, heightImg))
    cv2.imshow("img2", cropedImg2)
    cropedImg3 = cv2.resize(img_withowt_image, (widthImg, heightImg))
    cv2.imshow("img3", cropedImg3)
    return img_withowt_image,img_with_mood_boxes,img_with_wight_boxes,img_with_black_boxes,contours


#Fainalized vertical_separater
def vertical_separater(img,imgOrginal):

    imgCopy = img.copy()
    imgCopy2 = img.copy()
    sobelX = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    sobelY = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)


    sobelX = np.uint8(np.absolute(sobelX))

    sobelY = np.uint8(np.absolute(sobelY))

    # sobelX1 = cv2.cvtColor(sobelX, cv2.COLOR_BGR2GRAY)
    # sobelY1 = cv2.cvtColor(sobelY, cv2.COLOR_BGR2GRAY)

    ret3, sobelX1 = cv2.threshold(sobelX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # for vertical projection
    ret3, sobelY1 = cv2.threshold(sobelY, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # for vertical hough line ditection

    imgForVerticalLine_Inver_Canny = cv2.Canny(sobelY1, 10, 50, apertureSize=3)

    kernel = np.ones((1, 5))

    imgDialsobel = cv2.erode(imgForVerticalLine_Inver_Canny, kernel, iterations=2)  # APPLY DILATION
    kernel = np.ones((2, 5))
    imgDialsobel_D = cv2.dilate(imgDialsobel, kernel, iterations=1)  # APPLY DILATION


    lines_H = cv2.HoughLinesP(imgDialsobel_D, 1, np.pi / 2, 10, minLineLength=int(img.shape[0] * .04), maxLineGap=int(img.shape[0] * .009))

    kernel = np.ones((int(img.shape[0] / 550), int(img.shape[1] / 650)))
    sobelX1 = cv2.dilate(sobelX1, kernel, iterations=2)  # APPLY DILATION

    if (lines_H is not None):
        for line in lines_H:
            x1, y1, x2, y2 = line[0]
            cv2.line(sobelX1, (x1, y1), (x2, y2), (0, 0, 0), int(heightImg / 350))

    sobelX1 = cv2.resize(sobelX1, (widthImg, heightImg))  # RESIZE IMAGE
    _, sobelX1 = cv2.threshold(sobelX1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    sobelX1[sobelX1 == 0] = 1
    sobelX1[sobelX1 == 255] = 0
    blankImageforHorizontal = np.zeros((heightImg, widthImg, 3), np.uint8)
    horizontal_projection = np.sum(sobelX1, axis=1) * widthImg / heightImg
    print(horizontal_projection)
    for row in range(heightImg):
        cv2.line(blankImageforHorizontal, (0, row), (int(horizontal_projection[row]), row), (255, 255, 255), 1)

    cv2.rectangle(img=blankImageforHorizontal, pt1=(0, heightImg), pt2=(int((widthImg * widthImg * 0.98) / heightImg), 0), color=(0, 0, 0), thickness=-1)
    blankImageforHorizontal = cv2.cvtColor(blankImageforHorizontal, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.uint8)
    blankImageforHorizontal = cv2.dilate(blankImageforHorizontal, kernel, iterations=2)  # APPLY DILATION
    cv2.line(blankImageforHorizontal, (0, 0), (widthImg, 0), (0, 0, 0), 2)
    cv2.line(blankImageforHorizontal, (0, heightImg), (widthImg, heightImg), (0, 0, 0), 2)
    ret3, imgThresholdBWInvert_h = cv2.threshold(blankImageforHorizontal, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



    contours_h = cv2.findContours(blankImageforHorizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    contours_h = contours_h[0] if len(contours_h) == 2 else contours_h[1]

    rects = []

    # Just initialize bounding rects and set all bools to false
    for cnt in contours_h:
        rects.append(cv2.boundingRect(cnt))
    rects.sort(key=getXFromRecty)
    clusters_h = {}
    i = 0

    upperConer = {}
    bottomConer = {}
    imageArray = []
    a, b = 0, 0
    x1, y1 = 0, 0
    x, y, w, h = 0, 0, widthImg, heightImg


    for cntr in rects:
        x2, y2, w2, h2 = cntr[0], cntr[1], cntr[2], cntr[3]
        clusters_h[i] = [x2, y2, w2, h2]
        if (abs(y2) < 10):
            upperConer = [x2, y2, w2, h2]
            a = 1
        else:
            if (y2 + h2 > heightImg - 20):
                bottomConer = [x2, y2, w2, h2]
                b = 1
            else:

                if (h2 + y2 > heightImg * 0.25):
                    x, y, w, h = x2, y2, w2, h2
                    cropedImg = img[int(y1 * img.shape[0] / heightImg):int((y2 + (h2 / 2)) * img.shape[0] / heightImg),
                                0:int(widthImg * img.shape[1] / widthImg)]
                    # cropedImg2 = cv2.resize(cropedImg, (int(cropedImg.shape[1]*widthImg/img.shape[1]),int(cropedImg.shape[0]*heightImg/img.shape[0]) ))

                    x1 = x2
                    y1 = int(y2 + h2 / 2)

                    imageArray.append(cropedImg)

        # cv2.waitKey(10)
        i = i + 1
    if (heightImg - y1 > heightImg * 0.1):
        cropedImg = imgCopy[int(y1 * img.shape[0] / heightImg):img.shape[0], 0:img.shape[1]]

        imageArray.append(cropedImg)
    if (len(imageArray) == 0):
        imageArray.append(img)

    return imageArray


def blur_Sobel(img):  # DONT USE ADEPTIVE THRESH
    imgCopy1 = img.copy()
    imgCopy2 = img.copy()
    kernel = np.ones((1, 5))

    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    # sobelCombined = cv2.cvtColor(sobelCombined, cv2.COLOR_BGR2GRAY)

    ret3, imgThresholdBW = cv2.threshold(sobelCombined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = imgThresholdBW.copy()

    return bw, sobelCombined
def pixeldensity(img,imgOrginal,thresh):
    col_details=drowLines(img,imgOrginal,thresh)

    imgThresholdBW = blur_Sobel(img)[0]
    ret3, imgThresholdBWInvert = cv2.threshold(imgThresholdBW, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = imgThresholdBW.shape

    # kernel = np.ones((5, 1))
    # image_moprphology_verti = cv2.morphologyEx(imgThresholdBWInvert, cv2.MORPH_RECT, kernel)
    kernel = np.ones((2, 5))
    image_moprphology_hori = cv2.morphologyEx(imgThresholdBWInvert, cv2.MORPH_ERODE, kernel)

    cropdResize_imgThresholdBWInvert_h = cv2.resize(image_moprphology_hori, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("cropdResize_imgThresholdBWInvert_h_h", cropdResize_imgThresholdBWInvert_h)

    # cv2.imshow("image_moprphology_verti", image_moprphology_verti)
    # cv2.imshow("image_moprphology_hori", image_moprphology_hori)

    # image_moprphology_verti[image_moprphology_verti == 0] = 0
    # image_moprphology_verti[image_moprphology_verti == 255] = 1
    # # cv2.imshow("imgThresholdBW", blur_Sobel(img)[0])
    # vertical_projection = np.sum(image_moprphology_verti, axis=0) * width / heightImg

    image_moprphology_hori[image_moprphology_hori == 0] = 0
    image_moprphology_hori[image_moprphology_hori == 255] = 1
    # cv2.imshow("imgThresholdBW", blur_Sobel(img)[0])
    horizontal_projection = np.sum(image_moprphology_hori, axis=1) * width / height


    # print('width : ', width)
    # print('height : ', height)
    blankImageforHorizontal = np.zeros((height, width, 3), np.uint8)
    # blankImageforVerticle = np.zeros((height, width, 3), np.uint8)

    # for col in range(0, width):
    #     cv2.line(blankImageforVerticle, (col, height), (col, height - int(vertical_projection[col])),
    #              (255, 255, 255), 1)
    for row in range(height):
        cv2.line(blankImageforHorizontal, (0, row),
                 (int(horizontal_projection[row]), row),
                 (255, 255, 255), 1)
    cropdResize_imgThresholdBWInvert_h = cv2.resize(blankImageforHorizontal, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("cropdResize_imgThresholdBWInvert_h", cropdResize_imgThresholdBWInvert_h)

    # mean_vertical = int(np.mean(vertical_projection))
    # max_vertical = int(np.max(vertical_projection))
    max_horizontal = int(np.max(horizontal_projection))
    mean_horizontal = int(np.mean(horizontal_projection))
    # # print("M=",mean)
    # imgHistrogram = blankImage.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    # cv2.imshow("imgHistrogram", imgHistrogram)
    # cv2.rectangle(img=blankImageforVerticle, pt1=(0, height), pt2=(width, height - mean + 100), color=(0, 0, 0), thickness=-1 )
    cv2.rectangle(img=blankImageforHorizontal, pt1=(0, height),
                  pt2=(int((max_horizontal * 3 + mean_horizontal) / 4), 0), color=(0, 0, 0), thickness=-1)
    # cv2.rectangle(img=blankImageforVerticle, pt1=(0, height),
    #               pt2=(width, int((mean_vertical + max_vertical * 2) / 3) - 50), color=(0, 0, 0), thickness=-1)

    # cv2.imshow("blankImageforVerticle", blankImageforVerticle)
    # cv2.imshow("blankImageforHorizontal", blankImageforHorizontal)
    # # cv2.rectangle(blankImage, , ,, -1)

    # blankImageforVerticle = cv2.cvtColor(blankImageforVerticle, cv2.COLOR_BGR2GRAY)
    # ret3, imgThresholdBWInvert_v = cv2.threshold(blankImageforVerticle, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # kernel = np.ones((5, 5))
    # imgThresholdBWInvert_v = cv2.morphologyEx(imgThresholdBWInvert_v, cv2.MORPH_ERODE, kernel)
    # imgThreshold = cv2.Canny(imgThresholdBWInvert_v, thres[0], thres[1])  # APPLY CANNY BLUR
    # cv2.imshow("imgThreshold", imgThresholdBWInvert_v)
    # ret3, imgThresholdBWInvert_v = cv2.threshold(imgThresholdBWInvert_v, 0, 255,
    #                                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # contours_v = cv2.findContours(imgThresholdBWInvert_v, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # contours_v = contours_v[0] if len(contours_v) == 2 else contours_v[1]
    # rects_v=[]
    # sum_col_width=0
    # for cntr in contours_v:
    #     sum_col_width=sum_col_width+cv2.boundingRect(cntr)[2]
    #     rects_v.append(cv2.boundingRect(cntr))
    #
    # sum_col_width=sum_col_width*imgOrginal.shape[1]/ widthImg
    # rects_v.sort(key=getXFromRectx)
    # leftConer=0,0,0,0
    # rightConer=[int(width*(imgOrginal.shape[1]/widthImg)),0,0,0]
    # if len(rects_v)>1:
    #     if (rects_v[0][0] < 20):
    #         leftConer = [int(rects_v[0][0]*imgOrginal.shape[1]/widthImg),0,int(rects_v[0][2]*imgOrginal.shape[1]/widthImg),0]
    #     if(width-rects_v[len(rects_v)-1][0]-rects_v[len(rects_v)-1][2]<20 ):
    #         #rightConer =rects_v[len(rects_v)-1] **imgOrginal.imgThresholdBW.shape[1]/widthImg
    #         rightConer = [int(rects_v[len(rects_v)-1][0] * imgOrginal.shape[1] / widthImg), 0,
    #                      int(rects_v[len(rects_v)-1][2] * imgOrginal.shape[1] / widthImg), 0]
    leftConer=col_details[0][0]
    rightConer=col_details[0][1]
    upperConer=col_details[1][0]
    bottomConer=col_details[1][1]
    sum_col_width=0
    num_of_effective_col=len(col_details[3])
    imgCopy=img.copy()
    for col in col_details[3]:#collem separation lines
        x, y, w, h = col
        sum_col_width = sum_col_width +  w
        cv2.line(imgCopy, (int(x+(w/2)), 0),(int(x+(w/2)), imgCopy.shape[0]),(255, 255, 255), 5)



    if(leftConer[2]>0):
        num_of_effective_col=num_of_effective_col-1
    if(rightConer[2]>0):
        num_of_effective_col=num_of_effective_col-1

    median_col_width=(sum_col_width-leftConer[2]-rightConer[2])/1 if num_of_effective_col==0 else num_of_effective_col
    print("median",median_col_width)
    print("l:", leftConer)
    print("R:", rightConer)
    print("clusters:", col_details[3])

    # blankImageforHorizontal = cv2.cvtColor(blankImageforHorizontal, cv2.COLOR_BGR2GRAY)
    # ret3, imgThresholdBWInvert_h = cv2.threshold(blankImageforHorizontal, 0, 255,
    #                                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # kernel = np.ones((5, 5))
    # imgThresholdBWInvert_h = cv2.morphologyEx(imgThresholdBWInvert_h, cv2.MORPH_ERODE, kernel)
    # imgThreshold_h = cv2.Canny(imgThresholdBWInvert_h, thres[0], thres[1])  # APPLY CANNY BLUR
    # # cv2.imshow("imgThreshold_h", imgThresholdBWInvert_h)
    # ret3, imgThresholdBWInvert_h = cv2.threshold(imgThresholdBWInvert_h, 0, 255,
    #                                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #
    # contours_h = cv2.findContours(imgThresholdBWInvert_h, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # contours_h = contours_h[0] if len(contours_h) == 2 else contours_h[1]
    # rects_h=[]
    # for cntr in contours_h:
    #     rects_h.append(cv2.boundingRect(cntr))
    # rects_h.sort(key=getXFromRecty)
    # upperConer = 0, 0, 0, 0
    # bottomConer = [0,int(height * (imgOrginal.shape[0] / height)), 0, 0]
    # if len(rects_h) > 1:
    #     if (rects_h[0][1] < 20):
    #         upperConer = [0,int(rects_h[0][1] * imgOrginal.shape[0] / height),
    #                      0,int(rects_h[0][3] * imgOrginal.shape[0] / height),]
    #     if (height - rects_h[len(rects_h) - 1][1] - rects_h[len(rects_h) - 1][3] < 20):
    #         # rightConer =rects_v[len(rects_v)-1] **imgOrginal.imgThresholdBW.shape[1]/widthImg
    #         bottomConer = [0,int(rects_h[len(rects_h) - 1][1] * imgOrginal.shape[0] / height),
    #                       0,int(rects_h[len(rects_h) - 1][3] * imgOrginal.shape[0] / height) ]


    print("U:", upperConer)
    print("B:", bottomConer)
    rmv = remove_images(img,imgOrginal,thresh)
    imgWithoutPic=rmv[1].copy()
    cropedImg = imgWithoutPic[upperConer[1]+upperConer[3]-5:bottomConer[1], leftConer[0]+leftConer[2]:rightConer[0]]


    kernel = np.ones((2, 2))
    gradiant = cv2.morphologyEx(cropedImg, cv2.MORPH_GRADIENT, kernel)


    # imgAdaptiveThre = cv2.adaptiveThreshold(gradiant, 255, 1, 1, 7, 2)
    ret3, otsu = cv2.threshold(gradiant, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret3, otsu2 = cv2.threshold(cropedImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # karnalSize=1
    # if(int((otsu.shape[1]^4)/(widthImg^4)/2)>1):
    #     karnalSize=int((otsu.shape[1]^3)/(widthImg^3)/2)
    # kernel = np.ones((karnalSize,karnalSize))

    # otsu_After_D = cv2.dilate(otsu, kernel, iterations=3)  # APPLY DILATION

    cropdResize = cv2.resize(otsu, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("22", cropdResize)

    contours = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    rects = []
    blankImage = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    total_area=0
    for cntr in contours:

        x, y, w, h = cv2.boundingRect(cntr)
        total_area=total_area+ cv2.contourArea(cntr)
        #img2 = cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
    print("tt:",total_area/(otsu.shape[0]*otsu.shape[1]))

    contours = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    rects=[]
    for cntr in contours:
        rects.append(cv2.boundingRect(cntr))
    rects.sort(key=getXFromRectx)



    sum_of_vertical_gap=0
    sum_of_effective_conters=0
    median_col_width= (leftConer[2]+rightConer[2])/2 if median_col_width<1 else median_col_width
    for i in range(0, len(rects)):
        x1,y1,w1,h1=rects[i]
        for j in range(i+1, len(rects)):
            x2, y2, w2, h2 = rects[j]
            if (((y1 <= y2) and (y2 <= y1 + h1)) or ((y1 <= y2 + h2) and (y2 + h2 <= y1 + h1)) or (y2 <= y1 and (y1 + h1 <= y2 + h2))) and (abs(h1-h1)<max(h1,h2)*0.4):
                if ((x2-(x1+w1))>0) and ((x2-(x1+w1))<median_col_width):
                    sum_of_vertical_gap=sum_of_vertical_gap+x2-(x1+w1)
                    sum_of_effective_conters=sum_of_effective_conters+1
                    break
                else:
                    break

    #ave_of_vertical_gap=sum_of_vertical_gap/ sum_of_effective_conters
    #print("ave_of_vertical_gap",ave_of_vertical_gap)ss


def lineExtender(img,imgOrginal,thresh,col_details,rmv,line_d):#    col_details = drowLines(img, imgOrginal, thresh) rmv = remove_images(imgCopy1, imgCopy2, thresh)line_d=remove_text(rmv[1],imgOrginal,rmv[4])

    imgCopy=img.copy()
    imgCopy=cv2.cvtColor(imgCopy,cv2.COLOR_GRAY2RGB)
    imgCopy1=img.copy()
    imgCopy2=imgOrginal.copy()
    imgCopy3=imgCopy.copy()
    blankImage_x = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    blankImage_y = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)



    collems = col_details[3]
    leftConer = col_details[0][0]
    rightConer = col_details[0][1]
    upperConer = col_details[1][0]
    bottomConer = col_details[1][1]

    x_line=[]
    x_line_pointer=[]
    y_line=[]
    forgrount_line_y = blankImage_y.copy()
    contours = line_d[1]

    for cntr in contours:

        x, y, w, h = cntr
        if(((w/h)>8) & (w>img.shape[1]/50)):


            blankImage_x = cv2.line(blankImage_x, (x, y), (x + w, y), (255, 255, 255), 5)
            blankImage_x = cv2.line(blankImage_x, (x, y + h), (x + w, y + h), (255, 255, 255), 5)
            # imgCopy3 = cv2.line(imgCopy3, (x, y), (x + w, y), (255, 255, 255), 5)
            # imgCopy3 = cv2.line(imgCopy3, (x, y + h), (x + w, y + h), (255, 255, 255), 5)

        elif (((h/w)>8) & (h>img.shape[0]/50)):

            blankImage_y = cv2.line(blankImage_y, (x, y), (x , y+h), (255, 255, 255), 5)
            blankImage_y = cv2.line(blankImage_y, (x+w, y ), (x + w, y + h), (255, 255, 255), 5)
            forgrount_line_y = cv2.line(forgrount_line_y, (int(x+w/2), y ), (int(x+w/2), y + h), (255, 255, 255), int(forgrount_line_y.shape[0] / 500))
    blankImageResizeB = cv2.resize(blankImage_y, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("blankImage_y", blankImageResizeB)

    length_of_real_lines = line_d[2]

    # forgrount_line_y=blankImage_y.copy()
    # blankImageResizeB = cv2.resize(forgrount_line_y, (widthImg, heightImg))  # RESIZE IMAGE
    # cv2.imshow("blankImage_y", blankImageResizeB)
    lines_V=col_details[5]
    if (lines_V is not None):
        for line in lines_V:
            x1, y1, x2, y2 = line[0]
            cv2.line(forgrount_line_y, (x1, y1), (x2, y2), (255, 255, 255), int(forgrount_line_y.shape[0] / 500))#acurateed line set
    all_lines_y=forgrount_line_y.copy()
    all_lines_y_b = cv2.resize(forgrount_line_y, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("all_lines_y_b", all_lines_y_b)
    kernel = np.ones((2, 20))
    all_lines_y_imgDial = cv2.dilate(all_lines_y, kernel, iterations=1)  # APPLY DILATION
    all_lines_y_imgErod = cv2.erode(all_lines_y_imgDial, kernel, iterations=1)  # APPLY DILATION
    all_lines_y_imgDialR = cv2.resize(all_lines_y_imgErod, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("all_lines_y_imgDialR", all_lines_y_imgDialR)
    blankImage_x = cv2.rectangle(blankImage_x, (0, 0), (leftConer[0]+leftConer[2], blankImage_x.shape[0]), (0, 0, 0), -1)
    blankImage_x = cv2.rectangle(blankImage_x, (rightConer[0], 0), (rightConer[0]+rightConer[2], blankImage_x.shape[0]), (0, 0, 0), -1)
    blankImage_x = cv2.rectangle(blankImage_x, (0, 0), (blankImage_x.shape[1], upperConer[1]+upperConer[3]), (0, 0, 0), -1)
    blankImage_x = cv2.rectangle(blankImage_x, (0,bottomConer[1]), (blankImage_x.shape[1], bottomConer[1]+bottomConer[3]), (0, 0, 0), -1)

    blankImage_y = cv2.rectangle(blankImage_y, (0, 0), (leftConer[0]+leftConer[2], blankImage_y.shape[0]), (0, 0, 0), -1)
    blankImage_y = cv2.rectangle(blankImage_y, (rightConer[0], 0), (rightConer[0]+rightConer[2], blankImage_y.shape[0]), (0, 0, 0), -1)
    blankImage_y = cv2.rectangle(blankImage_y, (0, 0), (blankImage_y.shape[1], upperConer[1]+upperConer[3]), (0, 0, 0), -1)
    blankImage_y = cv2.rectangle(blankImage_y, (0,bottomConer[1]), (blankImage_y.shape[1], bottomConer[1]+bottomConer[3]), (0, 0, 0), -1)

    blankImageResizeA = cv2.resize(blankImage_y, (widthImg, heightImg))  # RESIZE IMAGE
    # cv2.imshow("A", blankImageResizeA)

    kernel = np.ones((2, 50))
    imgDialx = cv2.dilate(blankImage_x, kernel, iterations=2)  # APPLY DILATION
    imgDialx = cv2.erode(imgDialx, kernel, iterations=3)  # APPLY DILATION
    kernel = np.ones((10, 10))
    imgDialx = cv2.dilate(imgDialx, kernel, iterations=1)  # APPLY DILATION
    imgDialx = cv2.erode(imgDialx, kernel, iterations=1)  # APPLY DILATION
    ret3, otsu = cv2.threshold(cv2.cvtColor(imgDialx,cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    otsu=cv2.cvtColor(otsu,cv2.COLOR_GRAY2RGB)

    for cntr in contours:

        x, y, w, h = cv2.boundingRect(cntr)
        if (True):

            x_line.append([x, y, w, h,False])
            imgCopy=cv2.line(imgCopy, (x, int(y+(h/2))), (x+w, int(y+(h/2))), (0, 255 ,0), 30)#((x, ), (w, ))#horizontal real all lines

    kernel = np.ones((50, 2))
    imgDialy = cv2.dilate(blankImage_y, kernel, iterations=2)  # APPLY DILATION
    imgDialy = cv2.erode(imgDialy, kernel, iterations=2)  # APPLY DILATION
    kernel = np.ones((20, 20))
    imgDialy = cv2.dilate(imgDialy, kernel, iterations=1)  # APPLY DILATION
    imgDialy = cv2.erode(imgDialy, kernel, iterations=1)  # APPLY DILATION
    ret3, otsu_y = cv2.threshold(cv2.cvtColor(imgDialy, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_y = cv2.findContours(otsu_y, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours_y = contours_y[0] if len(contours_y) == 2 else contours_y[1]


    for cntr in contours_y:

        x, y, w, h = cv2.boundingRect(cntr)
        if (True):

            # y_line.append([x, y, w, h,False])
            # blankImage_y = cv2.line(imgCopy3, (int(x+(w/2)), y), (int(x+(w/2)), y+h), (0, 255 ,0),5)#((x, ), (w, ))
            imgCopy = cv2.line(imgCopy, (int(x+(w/2)), y), (int(x+(w/2)), y+h), (0, 255 ,0),5)#vertical real all lines
    blankImageResizeyyy = cv2.resize(blankImage_y, (widthImg, heightImg))  # RESIZE IMAGE
    # cv2.imshow("blankImageResizeyyy", blankImageResizeyyy)
    # y_line.sort(key=getXFromRectx)



    #################################################################################
    x_line.sort(key=getXFromRecty)
    i = 0
    j = 0
    maxWidth = 0
    maxWidthIndex=0
    for line in x_line:
        x1, y1, w1, h1, s1 = line


        if imgCopy.shape[0]/4 >y1+h1/2:
            if((maxWidth<w1) or abs(maxWidth-w1)<maxWidth/10):
                maxWidth=w1
                maxWidthIndex=int(y1+h1/2)
                j=i
        else:
            break
    i=i+1
    x_lineCopy =[]
    upperBoder=[[leftConer[0]+leftConer[2],upperConer[1]+upperConer[3]],[rightConer[0],upperConer[1]+upperConer[3]]]
    bottomBoder=[[leftConer[0]+leftConer[2],bottomConer[1]],[rightConer[0],bottomConer[1]]]

    # x_line=x_lineCopy.copy()
    isUpChage=False
    changingFactor=0
    if(maxWidth>imgCopy.shape[1]*3/5 or abs(maxWidth-imgCopy.shape[1]*3/5)<100 ):


        k=0
        changingFactor=abs( upperBoder[0][1] - maxWidthIndex)
        upperBoder[0][1] = maxWidthIndex
        upperBoder[1][1] = maxWidthIndex
        isUpChage=True
        print("ll:",length_of_real_lines / col_details[6])
        if ((length_of_real_lines / col_details[6]) > img.shape[0]  / 2):  # FOR remove near line for UP
            dy = 150 if (length_of_real_lines / col_details[6] )> (img.shape[0] *3 / 4) else 100
        else:
            dy = 250 if maxWidthIndex > img.shape[0] / 9 else 250

        for line in x_line:
            x1, y1, w1, h1, s1 = line

            if(y1<maxWidthIndex):
                continue

            if(int(y1+h1/2)-maxWidthIndex<dy):
                continue
            x_lineCopy.append([x1, y1, w1, h1, s1])
            k=k+1
        x_line=x_lineCopy.copy()
    if not isUpChage:

        for line in x_line:
            x1, y1, w1, h1, s1 = line

            dy= 100
            if abs(upperConer[1] + upperConer[3] - y1 - h1 / 2) < dy  :
                continue
            x_lineCopy.append([x1, y1, w1, h1, s1])
        x_line = x_lineCopy.copy()

    contours = line_d[1]
    y_lines_cross_ub=[]
    cv2.waitKey(10)
    lines_2 = []
    imgtest = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    dy=200
    if maxWidthIndex>img.shape[0]/9:
        dy = 250

    for cntr in contours:

        x, y, w, h = cntr

        if (((h / w) > 8) & (h > img.shape[0] / 50)):
            lines_2.append([x, y, x + w, y + h])
            imgtest = cv2.line(imgtest, (x, y), (x, y+h), (255, 0, 255), 10)

            if((y-100<=(upperBoder[0][1])) and ((upperBoder[0][1])<(y+h)) )|((y>=(upperBoder[0][1])) and ((upperBoder[0][1])>(y+h-100)) ):
                y_lines_cross_ub.append(cntr)
    lines_V = col_details[5]
    lines_=[]
    if  (length_of_real_lines/ col_details[6])<img.shape[0]*3/4:
        if (lines_V is not None):
            for line in lines_V:
                x1, y1, x2, y2 = line[0]
                lines_.append(line[0])
                imgtest = cv2.line(imgtest, (x1, y1+50), (x2, y2+50), (0, 0, 255), 10)

                if ((y1-50 <=( upperBoder[0][1] )) and ((upperBoder[0][1]) < y2))|((y1 >= (upperBoder[0][1] )) and (((upperBoder[0][1]) > y2-50))):
                    y_lines_cross_ub.append([x1,y1,abs(x1-x2),abs(y1-y2)])
    imgtest = cv2.line(imgtest, (upperBoder[0][0], upperBoder[0][1]), (upperBoder[1][0], upperBoder[1][1]), (0, 255, 0), 10)

    imgL = cv2.resize(imgtest, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("imgL", imgL)

    y_lines_cross_ub_collems=[]
    for cntr in y_lines_cross_ub:
        x, y, w, h = cntr
        minGap=imgtest.shape[1]
        near_col=cntr
        for col in collems:
            x2, y2, w2, h2 = col
            if abs(x2+(w2/2)-(x+(w/2)))<minGap:
                minGap=abs(x2+(w2/2)-(x+(w/2)))
                near_col= [ x2, upperBoder[0][1], w2, h2]


        if near_col not in y_lines_cross_ub_collems:
            y_lines_cross_ub_collems.append(near_col)
            imgtest = cv2.line(imgtest, (int(x + w / 2), y), (int(x + w / 2), y + h), (255, 255, 255), 20)
            imgtest = cv2.line(imgtest, (int(near_col[0] + near_col[2] / 2), y),(int(near_col[0] + near_col[2] / 2), y + h), (0, 0, 255), 10)

    if [leftConer[0], upperBoder[0][1],leftConer[2],0] not in y_lines_cross_ub_collems:
        y_lines_cross_ub_collems.append([leftConer[0], upperBoder[0][1],leftConer[2],0])
    if [rightConer[0],upperBoder[0][1],rightConer[2],0] not in y_lines_cross_ub_collems:
        y_lines_cross_ub_collems.append([rightConer[0],upperBoder[0][1],rightConer[2],0])
    y_lines_cross_ub_collems.sort(key=getXFromRectx)
    imgtest = cv2.line(imgtest, (upperBoder[0][0], upperBoder[0][1]), (imgtest.shape[1],upperBoder[0][1]), (255, 255, 255), 10)
    # imgtest = cv2.line(imgtest, (upperBoder[0][0], upperBoder[0][1]+ (200 if isUpChage else 200)), (imgtest.shape[1],upperBoder[0][1]+ (200 if isUpChage else 200)), (255, 0, 255), 5)
    # x=100

    # x_line.append([upperBoder[0][0], upperBoder[0][1], upperBoder[1][0]-upperBoder[0][0], upperBoder[1][1]- upperBoder[0][1], False])
    x_line.append([bottomBoder[0][0], bottomBoder[0][1], bottomBoder[1][0]-bottomBoder[0][0], bottomBoder[1][1]- bottomBoder[0][1], False])
    x_line.sort(key=getXFromRecty)
    extended_lines=[]

    for cntr in x_line:
        x1, y1, w1, h1,s1 =cntr

        leftcollem=collems[0][0]
        rightcollem=collems[len(collems)-1][0]


        if  (w1>imgCopy.shape[1]/100):
            for col in collems:
                x2, y2, w2, h2 = col

                if (int(x2)<x1):# | ( x2-x1>0 and x2-x1<w2*2/3 ):
                    leftcollem=int( x2+(w2/2))

                if (int( x2+w2)>x1+w1) :#or( x1-x2>0 and x1-x2<w2 ):
                    rightcollem=int( x2+(w2/2))
                    break

            extended_lines.append([leftcollem, int(y1 + (h1 / 2)),rightcollem, int(y1 + (h1 / 2)),True])
    extended_lines.sort(key=getXFromRecty)

    fx_lines=[]
    for i in range(0, len(extended_lines)):
        x1, y1, x2, y2,s = extended_lines[i]
        if(s):
            for j in range(i+1, len(extended_lines)):
                x_1, y_1, x_2, y_2,s_s = extended_lines[j]
                if(s_s):
                    if(abs((x2-x1)-(x_2-x_1))<10) & ((y_1-y1)<((imgCopy.shape[0]*abs(x2-x1))/(imgCopy.shape[1]*4))) & (abs(x1-x_1)<10):

                        extended_lines[j][4]=False
                    elif((y_1-y1)<imgCopy.shape[0]/80):
                        if(abs( x1- x_1)<10)|(abs( x2- x_2)<10):
                            if((x2-x1)>(x_2-x_1)):
                                extended_lines[j][4] = False
                            else:
                                extended_lines[i][4] = False
                        if((x1<x_1)&(x_2<x2))|((x1>x_1)&(x_2>x2)):
                            if ((x2 - x1) > (x_2 - x_1)):
                                extended_lines[j][4] = False
                            else:
                                extended_lines[i][4] = False
    z=0
    # for i in range(0, len(y_lines_cross_ub_collems) - 1):
    #     x1, y1, w1, h1 = y_lines_cross_ub_collems[i]
    #     x2, y2, w2, h2 = y_lines_cross_ub_collems[i + 1]
    #     # imgCopy = cv2.line(imgCopy, (x1, y1), (x1, 500), (255, 0, 255), 10)  # ((x, ), (w, ))
    #     # imgCopy = cv2.line(imgCopy, (int(x1 + w1 / 2), 500), (int(x2 + w2 / 2) - int(x1 + w1 / 2), 500), (255, 0, 255),
    #     #                    10)  # ((x, ), (w, ))
    #
    #     extended_lines.append([int(x1 + w1 / 2), 10, int(x2 + w2 / 2) - int(x1 + w1 / 2), 5, False])
    x=100
    for i in range(0, len(y_lines_cross_ub_collems)-1):
        x1, y1, w1, h1 = y_lines_cross_ub_collems[i]
        x2, y2, w2, h2 = y_lines_cross_ub_collems[i+1]
        # imgCopy = cv2.line(imgCopy, (x1, y1), (x1,250), (x, x, 0), 10)  # ((x, ), (w, ))
        # imgCopy = cv2.line(imgCopy, (int(x1+w1/2)+50, 500), (int(x2+w2/2)-50,500), (x, 0, x), 10)  # ((x, ), (w, ))
        x=x+50
    #
        extended_lines.append([int(x1+w1/2), upperBoder[0][1],(int(x2+w2/2)), upperBoder[0][1], True])
    extended_lines.sort(key=getXFromRecty)
    for i in range( len(extended_lines)-1,-1,-1):
        x1, y1, x2, y2, s = extended_lines[i]

        if (s):
            l_bottom = (x1, imgCopy.shape[0])
            r_bottom = (x2, imgCopy.shape[0])
            for j in range(i+1, len(extended_lines)):
                x_1, y_1, x_2, y_2, s_s = extended_lines[j]
                if (s_s):
                    if(x1==x_1)|((x_1<x1) &(x1<x_2)):
                        l_bottom=(x1,y_1)
                        break
            for j in range(i+1, len(extended_lines)):
                x_1, y_1, x_2, y_2, s_s = extended_lines[j]
                if (s_s):
                    if(x2==x_2) |((x_1<x2) &(x2<x_2)):
                        r_bottom=(x2,y_2)
                        break
            imgCopy = cv2.line(imgCopy, (x1, y1), l_bottom, (0, 0, 255), 10)  # ((x, ), (w, ))
            imgCopy = cv2.line(imgCopy, (x2, y2), r_bottom, (0, 0, 255), 10)  # ((x, ), (w, ))



    for line in extended_lines:
        x1, y1, x2, y2, s = line
        if(s):
            blankImage_x = cv2.line(imgCopy, (x1, y1), (x2, y2), (0, 0,255), 10)  # ((x, ), (w, ))





    cv2.imwrite("blankImage_x.jpg",blankImage_x)
    blankImageResizey = cv2.resize(imgCopy, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("textRMV_Rx111", blankImageResizey)
    blankImageResizey2 = cv2.resize(imgtest, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("textRMV_Rx222", blankImageResizey2)


    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.waitKey(0)

while True:

    img = cv2.imread("o19.jpg")



    # row, col = im.shape[:2]
    # bottom = im[row - 2:row, 0:col]1
    # mean = cv2.mean(bottom)[0]
    #
    # bordersize = 50
    # border = cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
    #                             borderType=cv2.BORDER_CONSTANT, value=[mean, mean, mean])

    # border = cv2.resize(border, (widthImg, heightImg))

    # img = cv2.resize(img, (s, heightImg))  # RESIZE IMAGE
    imgBlank = np.zeros((img.shape[0], img.shape[1], 3),
                        np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    thres = utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=3)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=2)  # APPLY EROSION

    cv2.imshow("imgWarpColored", cv2.resize(imgThreshold, (widthImg, heightImg)))
    ## FIND ALL COUNTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 1)  # DRAW ALL DETECTED CONTOURS
    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32(
            [[0, 0], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (img.shape[0], img.shape[1]))
        # REMOVE 10 PIXELS FORM EACH SIDE
        imgWarpColored = imgWarpColored[5:imgWarpColored.shape[0] - 10, 5:imgWarpColored.shape[1] - 10]
        imgWarpColored = cv2.resize(imgWarpColored, (img.shape[1], img.shape[0]))
        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)
        # Image Array for Display
        imageArray = ([img, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray])
    else:
        imageArray = ([img, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank])
        imgWarpGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # LABELS FOR DISPLAY
    lables = (["Original", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Prespective", "Warp Gray"])
    # stackedImage = utlis.stackImages(imageArray,0.75,lables)
    # cv2.imshow("Result",stackedImage)###########################################all results
    cv2.imshow("imgWarpColored", cv2.resize(imgWarpGray, (widthImg, heightImg)))
    cv2.waitKey(1)
    # pixeldensity(imgWarpGray)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('imgWarpGrayss.jpg', imgWarpGray)
        cv2.destroyAllWindows()
        utlis.initializeTrackbars()
        cv2.imshow("imgWarpColored", cv2.resize(imgWarpGray, (widthImg, heightImg)))
        #vertical_separater(imgWarpGray)
        imgArr = vertical_separater(cv2.resize(imgWarpGray, (widthImg, heightImg)),imgWarpGray)
        cv2.waitKey(100)
        i = 0
        ret4, imgThresholdBWInvert = cv2.threshold(imgWarpGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # pixeldensity(cv2.resize(imgWarpGray, (widthImg, heightImg)),imgWarpGray )
        imgCopyforRemoveImage1=imgWarpGray.copy()

        imgCopyforRemoveImage2=imgWarpGray.copy()

        #remove_images(imgCopyforRemoveImage1,imgCopyforRemoveImage1,ret4)
        # pixeldensity(imgCopyforRemoveImage1, imgCopyforRemoveImage1,ret4)
        rmv = remove_images(imgCopyforRemoveImage1, imgCopyforRemoveImage1, ret4)
        col_details = drowLines(imgCopyforRemoveImage1, imgCopyforRemoveImage1, ret4,rmv)
        line_d=remove_text(rmv[1],imgCopyforRemoveImage1,rmv[4])

        imgtest=lineExtender(imgCopyforRemoveImage1,imgCopyforRemoveImage1,ret4,col_details,rmv,line_d)
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)
        for CropedImg in imgArr:
            CropedImgCopy=CropedImg.copy()
            cropedImg2 = cv2.resize(CropedImg, (int(CropedImg.shape[1]*widthImg/img.shape[1]),int(CropedImg.shape[0]*heightImg/img.shape[0]) ))
            cv2.imshow(str(i),cropedImg2)

            #imgWithoutPictures=img_without_pic=remove_images(CropedImg,imgWarpGray,ret4)
            #cv2.imshow(str(i)+"ff",CropedImgCopy)

            #remove_text(CropedImgCopy,imgWarpGray,0)
            #drowLines(CropedImgCopy,imgWarpGray)

            i=i+1
            cv2.waitKey(100)

        cv2.waitKey(0)

        break

#test