import cv2
import numpy as np
import utlis
from pythonRLSA import rlsa
import random


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

def remove_text(img,imgOrginal):
    img2=cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2RGB)
    #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #sobelX = cv2.Sobel(img, cv2.CV_8U, 1, 0)
    #sobelY = cv2.Sobel(img, cv2.CV_8U, 0, 1)
    #sobelCombined = cv2.bitwise_and(sobelX, sobelY)
    kernel = np.ones((2,2))
    gradiant = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    rect = cv2.morphologyEx(gradiant, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((10, 0))
    dialate = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)


    #imgAdaptiveThre = cv2.adaptiveThreshold(gradiant, 255, 1, 1, 7, 2)
    ret3, otsu = cv2.threshold(gradiant, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("gradiant.jpg",otsu)
    #bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    # cv2.imshow("bw",bw)
    otsuCanny=cv2.Canny(otsu, 50, 150, apertureSize=3)

    contours = cv2.findContours(otsuCanny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    rects = []
    blankImage = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    blankImage2 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    for cntr in contours:

        x, y, w, h = cv2.boundingRect(cntr)
        if(((w/h)>8) & (w>50)):

            img2=cv2.line(img2, (x, y), (x+w, y ), (0, 255, 0), 5)
            img2=cv2.line(img2, (x, y+h), (x+w, y+h ), (0, 255, 0), 5)
            blankImage=cv2.line(blankImage, (x, y), (x+w, y ), (255, 255, 255), 5)
            blankImage=cv2.line(blankImage, (x, y+h), (x+w, y+h ), (255, 255, 255), 5)
        elif (((h/w)>8) & (h>50)):
            img2 = cv2.line(img2, (x, y), (x , y+h), (0, 255, 0), 5)
            img2 = cv2.line(img2, (x+w, y ), (x + w, y + h), (0, 255, 0), 5)
            blankImage = cv2.line(blankImage, (x, y), (x , y+h), (255, 255, 255), 5)
            blankImage = cv2.line(blankImage, (x+w, y ), (x + w, y + h), (255, 255, 255), 5)

        else:
            img2 = cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), -1)
    blankImage = cv2.resize(img2, (widthImg, heightImg))  # RESIZE IMAGE
    cv2.imshow("textRMV_R", blankImage)
    kernel = np.ones((5, 5))
    dilate = cv2.dilate(blankImage, kernel, iterations=3)  # APPLY DILATION
    dilate = cv2.cvtColor(dilate,cv2.COLOR_RGB2GRAY)
    otsuCanny = cv2.Canny(dilate, 50, 150, apertureSize=3)
    lines_H = cv2.HoughLinesP(otsuCanny, 1, np.pi / 360, 200, minLineLength=int(img.shape[0] * .008), maxLineGap=int(img.shape[0] * .05))
    # if (lines_H is not None):
    #     for line in lines_H:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(blankImage2, (x1, y1), (x2, y2), (255, 255, 0), 5)
    #
    #         if(y1==y2):
    #             lx=min(x1,x2)
    #             rx=max(x1,x2)
    #             nearLCol=0
    #             nearRCol=collems[0]
    #             i=0
    #             for col in len(collems):#collems must be sorted
    #                 if lx > col:
    #                   nearLCol=col
    #                 else:
    #                     break
    #             for col in len(collems):#collems must be sorted
    #                 if rx > col:
    #                     continue
    #                 else:
    #                     nearRCol = col
    #                     break




    #

    #resize = cv2.resize(imgAdaptiveThre, (widthImg, heightImg))  # RESIZE IMAGE
    # imgForHorizontalLine_Inver_r = cv2.resize(blankImage2, (widthImg, heightImg))  # RESIZE IMAGE
    # imgForVerticalLine_Invert_R = cv2.resize(otsuCanny, (widthImg, heightImg))  # RESIZE IMAGE

    #cv2.imshow("resize",resize)
    # cv2.imshow("gradiant",imgForHorizontalLine_Inver_r)
    # cv2.imshow("imgForVerticalLine_Invert_R",imgForVerticalLine_Invert_R)



    return img2,contours


def drowLines(img,imgOriginal,thresh):
    imgCopy1 = img.copy()
    imgCopy2 = img.copy()
    rmv=remove_images(imgCopy1,imgCopy2,thresh)

    img_withowt_pics=rmv[3]
    img_Conters=rmv[4]
    textRMV=remove_text(rmv[1], imgOriginal)
    contours=textRMV[1]


    kernel = np.ones((1, 5))

    sobelX = cv2.Sobel(img_withowt_pics, cv2.CV_16U, 1, 0)
    sobelY = cv2.Sobel(img_withowt_pics, cv2.CV_16U, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    for cntr in contours:

        x, y, w, h = cv2.boundingRect(cntr)
        if(((w/h)>8) & (w>50)):

            sobelY=cv2.line(sobelY, (x, y), (x+w, y ),(255, 255, 255) , 5)

            sobelX=cv2.line(sobelX, (x, y), (x+w, y ),(0, 0, 0) , 5)

        elif (((h/w)>8) & (h>50)):
            sobelY=cv2.line(sobelY, (x, y), (x+w, y ),(0, 0, 0), 5)
            sobelY=cv2.line(sobelY, (x+w, y ), (x + w, y + h), (0, 0, 0), 5)

            sobelX=cv2.line(sobelX, (x, y), (x+w, y ),(255, 255, 255), 5)
            sobelX=cv2.line(sobelX, (x+w, y ), (x + w, y + h), (0, 255, 255), 5)

    kernel = np.ones((1, 3))
    erodeY = cv2.erode(sobelY, kernel, iterations=1)  # APPLY DILATION
    imgDial = cv2.dilate(erodeY, kernel, iterations=1)  # APPLY DILATION

    kernelX = np.ones((3, 1))
    erodeX = cv2.erode(sobelX, kernelX, iterations=1)  # APPLY DILATION
    imgDialX = cv2.dilate(erodeX, kernelX, iterations=1)  # APPLY DILATION

    ret3, imgDial_I = cv2.threshold(imgDial, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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



    # imgDial_IR = cv2.resize(imgDial_v, (widthImg, heightImg))  # RESIZE IMAGE
    # cv2.imshow("imgDial_IR", imgDial_IR)

    kernel = np.ones((5, 5))

    image_rlsa_hori_dilate = cv2.erode(image_rlsa_hori, kernel, iterations=1)  # APPLY DILATION
    kernelx = np.ones((15, 2))
    image_rlsa_x_dilate = cv2.erode(image_rlsa_X, kernelx, iterations=1)  # APPLY DILATION



    lines_V = cv2.HoughLinesP(image_rlsa_hori_dilate, 5, np.pi, 100, minLineLength=int(image_rlsa_hori_dilate.shape[0] /4), maxLineGap=int(imgOriginal.shape[0] * .009))
    blankImage = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)



    if (lines_V is not None):
        for line in lines_V:
            x1, y1, x2, y2 = line[0]
            cv2.line(blankImage, (x1, y1), (x2, y2), (255, 255, 255), int(heightImg / 350))


    # textRMV_R = cv2.resize(blankImage, (widthImg, heightImg))  # RESIZE IMAGE


    kernel = np.ones((5, 5))
    blankImage = cv2.resize(blankImage, (widthImg, heightImg))  # RESIZE IMAGE
    blankImageX = cv2.resize(image_rlsa_x_dilate, (widthImg, heightImg))  # RESIZE IMAGE
    blankImage = cv2.dilate(blankImage, kernel, iterations=2)  # APPLY EROSION

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
    cv2.imshow("blankImageforVerticle",blankImageforVerticle)
    cv2.imshow("blankImageforX",blankImageforX)



    v_projection_C=blankImageforVerticle.copy()
    x_projection_C=blankImageforX.copy()
    blankImageforHorizontal=cv2.rectangle(img=x_projection_C, pt1=(0, heightImg), pt2=(int((widthImg * widthImg * 0.98) / heightImg), 0), color=(0, 0, 0), thickness=-1)
    blankImageforHorizontal = cv2.cvtColor(blankImageforHorizontal, cv2.COLOR_BGR2GRAY)
    #kernel = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.uint8)
    kernel = np.ones((3, 2))
    blankImageforHorizontal = cv2.dilate(blankImageforHorizontal, kernel, iterations=2)  # APPLY DILATION

    cv2.imshow("blankImageforHorizontals", blankImageforHorizontal)

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
        if (rects_h[0][1] < 20):
            upperConer = [0, rects_h[0][1],0, rects_h[0][3]  ]
        if (img.shape[0] - rects_h[len(rects_h) - 1][1] - rects_h[len(rects_h) - 1][3] <img.shape[0]/ 10):
            # rightConer =rects_v[len(rects_v)-1] **imgOrginal.imgThresholdBW.shape[1]/widthImg
            bottomConer =[0, rects_h[len(rects_h) - 1][1],0, rects_h[len(rects_h) - 1][3]  ]

    kernel = np.ones((5, 5))
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

        # img2=cv2.rectangle(blankImageforVerticle, (x, y), (x + w, y + h), (255, 0, 0), 1)

        if( (w > maxWidth) and (x != 0) and ((x + w) != widthImg)) |((w > maxWidth) and (w > widthImg/5)):
            maxWidth = w

    ###############################################################
    ##############################################################
    kernel = np.ones((10, 10))
    blankImageforVerticle_2 = cv2.dilate(blankImageforVerticle_2, kernel, iterations=1)  # APPLY EROSION
    contours_v = cv2.findContours(blankImageforVerticle_2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours_v = contours_v[0] if len(contours_v) == 2 else contours_v[1]

    for cntr in contours_v:
        x, y, w, h = cv2.boundingRect(cntr)

        v_projection_C=cv2.rectangle(v_projection_C, (x, y+20), (x + w, y + h), (0,0,0), -1)
        # img_with_wight_boxes=cv2.rectangle(img_with_wight_boxes, (x, y), (x + w, y + h), (255, 255, 255), -1)

    kernel = np.ones((5, 5))
    v_projection_C = cv2.dilate(v_projection_C, kernel, iterations=1)  # APPLY EROSION
    v_projection_C=cv2.cvtColor(v_projection_C,cv2.COLOR_RGB2GRAY)
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
        if (rects[0][0] < 20):
            leftConer = [rects[0][0] , rects[0][1],
                         rects[0][2] , 0]
        if (widthImg - rects[len(rects) - 1][0] - rects[len(rects) - 1][2] < 20):
            # rightConer =rects_v[len(rects_v)-1] **imgOrginal.imgThresholdBW.shape[1]/widthImg
            rightConer = [rects[len(rects) - 1][0] , rects[len(rects) - 1][1],
                          rects[len(rects) - 1][2] , 0]


    blankImage = cv2.resize(v_projection_C, (widthImg, heightImg))  # RESIZE IMAGE

    cv2.imshow("blankImageforVerticle11", blankImageforHorizontal)
    return [leftConer,rightConer],[upperConer,bottomConer],maxWidth,rects,rects_h


def remove_images(img,imgOrginal,tresh):

    kernalSize=int( img.shape[1] / widthImg/20)
    if ( kernalSize%2==0):kernalSize=kernalSize+1
    CropedImg1 = cv2.GaussianBlur(img,(kernalSize , kernalSize), 0)  # gaussian
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel = np.ones((10, 10))
    # grayC=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
    imgDial = cv2.dilate(CropedImg1, kernel, iterations=3)  # APPLY DILATION
    #cv2.imshow("dilate", imgDial)
    kernel = np.ones((5, 5))
    imgThreshold = cv2.erode(imgDial, kernel, iterations=2)  # APPLY EROSION
    imgThreshold = cv2.GaussianBlur(imgThreshold,(kernalSize , kernalSize), 0)  # gaussian

    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(imgThreshold, -1, kernel)  # applying the sharpening kernel to the input image & displaying it.
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
                ((x+w) == img.shape[1]) & ((y+h) == img.shape[0]))) | (h * w < imgOrginal.shape[0] * imgOrginal.shape[1] * 0.0009):

            continue
        #img_with_mood_boxes=cv2.rectangle(img_with_mood_boxes, (x, y), (x + w, y + h), (mood[0], mood[0], mood[0]), -1)
        img_with_mood_boxes=cv2.rectangle(img_with_mood_boxes, (x-8 if x>=8 else 0, y-10 if y>=10 else 0), (x+w+8 if x+w+8<=img_with_mood_boxes.shape[1] else img_with_mood_boxes.shape[1], y + h+10  if x+w+10<=img_with_mood_boxes.shape[0] else img_with_mood_boxes.shape[0]), (mood[0], mood[0], mood[0]), -1)
        img_with_black_boxes=cv2.rectangle(img_with_black_boxes, (x, y), (x + w, y + h), (0,0, 0), -1)
        img_with_wight_boxes=cv2.rectangle(img_with_wight_boxes, (x, y), (x + w, y + h), (255, 255, 255), -1)

        img_withowt_image=cv2.drawContours(img_withowt_image, [cntr], -1, (mood[0], mood[0], mood[0]), -1)


    # cropedImg2 = cv2.resize(img2, ( widthImg ,  heightImg ))

    # cv2.imshow("img2",cropedImg2)
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

def lineExtender(img,imgOrginal,thresh):
    print(1)

while True:

    img = cv2.imread("o7.jpg")

    # row, col = im.shape[:2]
    # bottom = im[row - 2:row, 0:col]
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

        #remove_images(imgCopyforRemoveImage1,imgCopyforRemoveImage1,100)
        # pixeldensity(imgCopyforRemoveImage1, imgCopyforRemoveImage1,ret4)
        imgtest=remove_text(imgCopyforRemoveImage1, imgCopyforRemoveImage1)
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