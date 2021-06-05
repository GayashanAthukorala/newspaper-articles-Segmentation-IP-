import cv2
import numpy as np
import utlis
from pythonRLSA import rlsa
import random
import matplotlib.pyplot as plt
from matplotlib import gridspec

########################################################################

cap = cv2.VideoCapture(0)
cap.set(2, 640)
cap.set(4, 480)
cap.set(10, 100)
heightImg = 640
widthImg = 480
utlis.initializeTrackbars()


def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

def mergCounters(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)
# Sort bounding rects by x coordinate
def getXFromRectx(item):
    return item[0]
def getXFromRecty(item):
    return item[1]

def merge_taitels(boundryArry):
    z = 0
    newboundryarry = {}
    for i in range(0, len(boundryArry)):
        x1, y1, w1, h1, s1 = boundryArry[i][0], boundryArry[i][1], boundryArry[i][2], boundryArry[i][3], boundryArry[i][
            4]
        if (s1 == 99): continue
        if ((h1 > (heightImg / 28))):
            boundryArry[i][4] = 99
            # cv2.rectangle(grayCopy3, (x3, y3), (x3 + w3, y3 + h3), (c1, c2, c3), 2)
            newboundryarry[z] = [x1, y1, w1, h1, 1]
            z = z + 1
            continue

        for j in range(i + 1, len(boundryArry)):
            x2, y2, w2, h2, s2 = boundryArry[j][0], boundryArry[j][1], boundryArry[j][2], boundryArry[j][3], \
                                 boundryArry[j][4]
            if (w2 * h2 < heightImg * widthImg / 3000): continue

            c1 = random.randint(0, 255)
            c2 = random.randint(0, 255)
            c3 = random.randint(0, 255)

            if ((abs(y2 - y1) < 5) | (abs(y2 + h2 - y1 - h1) < 5)) & (abs(h1 - h2) < 5) & (
            ((abs(x1 + w1 - x2) < h1 / 2) | (abs(x2 + w2 - x1) < h1 / 2))):

                print(x1, y1, w1, h1)
                print(x2, y2, w2, h2)
                # cv2.rectangle(grayCopy3, (x1, y1), (x1 + w1, y1 + h1), (c1,c2,c3), 2)
                # cv2.rectangle(grayCopy3, (x2, y2), (x2 + w2, y2 + h2), (c1,c2,c3), 2)

                x3, y3 = min(x1, x2), y2
                if (x1 > x2) & (x2 + w2 < x1 + w1):
                    w3 = x1 - x2 + w1
                if (x1 > x2) & (x2 + w2 > x1 + w1):
                    w3 = w2
                if (x1 < x2) & (x2 + w2 > x1 + w1):
                    w3 = x2 - x1 + w2
                if ((x1 < x2) & (x2 + w2 < x1 + w1)):
                    w3 = w1
                if (y2 < y1):
                    h3 = y1 - y2 + h1
                if (y2 + h2 > y1 + h1):
                    h3 = h2
                #

                print("seted")
                boundryArry[j][4] = 99
                # cv2.rectangle(grayCopy3, (x3, y3), (x3 + w3, y3 + h3), (c1, c2, c3), 2)
                # cv2.imshow("result5", grayCopy3)
                # cv2.waitKey(1000)
                x1, y1, w1, h1 = x3, y3, w3, h3
            #####################
        newboundryarry[z] = [x1, y1, w1, h1, 1]
        # print("ADED", x3, y3, w3, h3)
        z = z + 1

    return newboundryarry

def pixeldensity(img):
    imgThresholdBW = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 10)
    ret3, imgThresholdBWInvert = cv2.threshold(imgThresholdBW, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    imgThresholdBW[imgThresholdBW == 0] = 0
    imgThresholdBW[imgThresholdBW == 255] = 1
    #cv2.imshow("imgThresholdBW",imgThresholdBW)
    vertical_projection = np.sum(imgThresholdBW, axis=0);
    print(vertical_projection);
    height, width = imgThresholdBW.shape
    print('width : ', width)
    print('height : ', height)
    blankImage = np.zeros((height, width, 3), np.uint8)
    for row in range(0,width):
        cv2.line(blankImage, (row,height), (row,height-int(vertical_projection[row] * width / height)), (255, 255, 255), 1)
    mean=int(np.mean(vertical_projection))
    #print("M=",mean)
    imgHistrogram = blankImage.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    #cv2.imshow("imgHistrogram",imgHistrogram)
    cv2.rectangle(img=blankImage,pt1=(0,height),pt2=(width,height- mean+100),color=(0,0,0),thickness=-1,)
    #cv2.rectangle(blankImage, , ,, -1)
    blankImage = cv2.cvtColor(blankImage, cv2.COLOR_BGR2GRAY)
    ret3, imgThresholdBWInvert = cv2.threshold(blankImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    imgThreshold = cv2.Canny(imgThresholdBWInvert, thres[0], thres[1])  # APPLY CANNY BLUR
    #cv2.imshow("imgThreshold",imgThreshold)

    contours = cv2.findContours(blankImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    clusters={}
    i=0
    leftConer = {}
    rightConer = {}
    paraGapSize=0
    maxHight=0
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        clusters[i]=[x,y,w,h]
        if(abs(x-0)<5):
            leftConer=[x,y,w,h]
        else:
            if(x+w>width-5):
             rightConer=[x,y,w,h]
            else :
                if(maxHight<h):
                    maxHight=h
                    paraGapSize=w
        i=i+1

    print("l:",leftConer)
    print("R:",rightConer)
    print("clusters:",len(clusters))
    #if(len(clusters)>9):
    #Horizontal projection
    kernel = np.ones((1, 5))
    # lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    # lap = np.uint8(np.absolute(lap))

    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
  # sobelCombined = cv2.cvtColor(sobelCombined, cv2.COLOR_BGR2GRAY)

    ret3, imgThresholdBW = cv2.threshold(sobelCombined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    imgDialsobel = cv2.dilate(imgThresholdBW, kernel, iterations=1)  # APPLY DILATION
    imgThresholdsobel = cv2.erode(imgDialsobel, kernel, iterations=1)  # APPLY EROSION

    #cv2.imshow("imgThresholdsobel", imgThresholdsobel)

    horizontalical_projection = np.sum(imgThresholdsobel, axis=1)
    #print(horizontalical_projection);
    blankImageForHorizontalHistrogram= np.zeros((height, width, 3), np.uint8)
    for row in range( height):
        cv2.line(blankImageForHorizontalHistrogram, (0,row), (int(horizontalical_projection[row] * width / height),row),
        (255, 255, 255), 1)
    mean = int(np.mean(horizontalical_projection))
    # print("M=",mean)
    imgHistrogramHorizontal = blankImageForHorizontalHistrogram.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    #cv2.imshow("imgHistrogramHorizontal", imgHistrogramHorizontal)
        # cv2.rectangle(img=blankImageForHorizontalHistrogram, pt1=(0, 0), pt2=(width - mean + 100,0), color=(0, 0, 0),
        #               thickness=-1, )
        # blankImageForHorizontalHistrogram = cv2.cvtColor(blankImageForHorizontalHistrogram, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("blankImageForHorizontalHistrogram",blankImageForHorizontalHistrogram)
        #pixeldensity(img)
    print(clusters)



    #cv2.imshow("blankImage",blankImage)
    #cv2.imshow("orginal",img)

    #img=   imgThresholdBW
    # fig = plt.figure(figsize=(6, 6))
    # t = np.arange(img.shape[0])
    # f = np.arange(img.shape[1])
    # flim = (f.min(), f.max())
    # tlim = (t.min(), t.max())
    #
    # gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5])
    # gs.update(hspace=0, wspace=0)
    # ax = fig.add_subplot(gs[1, 0])
    # axl = fig.add_subplot(gs[1, 1], sharey=ax)
    # axb = fig.add_subplot(gs[0, 0], sharex=ax)
    # plt.setp(axl.get_yticklabels(), visible=False)
    # plt.setp(axb.get_xticklabels(), visible=False)
    # plt.setp(axl.get_xticklabels(), visible=False)
    # plt.setp(axb.get_yticklabels(), visible=False)
    #
    # ax.imshow(img, origin='lower', aspect='equal')
    #
    # axl.fill_between(img.mean(1), f)
    # axb.fill_between(t, img.mean(0))
    # ax.set_xlim(tlim)
    # ax.set_ylim(tlim)
    # # #imgThresholdBW = cv2.threshold(imgThresholdBW, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # # #ret3, imgThresholdBWInvert = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # # cv2.imshow("adeptiv",imgThresholdBWInvert)
    # # # cv2.imshow("otsu",imgThresholdBWInvert)
    # # histogram = {}  # Dictionary keeps count of different kinds of pixels in image
    # # histogram2 = {}  # Dictionary keeps count of different kinds of pixels in image
    # #
    # # for i in range(0, imgThresholdBWInvert.shape[0]):
    # #     pixels=0
    # #     for j in range(0, imgThresholdBWInvert.shape[1]):
    # #         pixels = pixels + imgThresholdBWInvert.item(i, j)
    # #     histogram[i]=float(pixels)
    # #     histogram2[i]=i
    # # print(histogram)
    # # a = np.histogram([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
    # # plt.hist(a, bins=histogram2)
    # # plt.title("histogram")
    # # plt.show()
    # #         # if pixel in histogram:
    # #         #     histogram[pixel] += 1  # Increment count
    # #         # else:
    # #         #     histogram[pixel] = 1  # pixel_val encountered for the first time
    return paraGapSize

def verticalSeparater(img):
    imgCopy=img.copy();
    imgCopy2=img.copy();
    kernel = np.ones((1, 5))
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    # setting the minimum and maximum thresholds
    # for double thresholding
    mag, ang = cv2.cartToPolar(sobelX, sobelX, angleInDegrees=True)
    mag_max = np.max(mag)
    iweak_th = mag_max * 0.1
    strong_th = mag_max * 0.5

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelX1 = cv2.cvtColor(sobelX, cv2.COLOR_BGR2GRAY)
    sobelY1 = cv2.cvtColor(sobelY, cv2.COLOR_BGR2GRAY)

    ret3, sobelX1 = cv2.threshold(sobelX1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#for vertical projection
    ret3, sobelY1 = cv2.threshold(sobelY1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#for vertical hough line ditection



    imgForVerticalLine_Inver_Canny = cv2.Canny(sobelY1, iweak_th, strong_th, apertureSize=3)

    kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    imgDialsobel = cv2.erode(imgForVerticalLine_Inver_Canny, kernel, iterations=2)  # APPLY DILATION

    kernel = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.uint8)
    imgDialsobel = cv2.erode(imgDialsobel, kernel, iterations=1)  # APPLY DILATION

    kernel = np.array([[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]], dtype=np.uint8)
    imgDialsobel = cv2.erode(imgDialsobel, kernel, iterations=1)  # APPLY DILATION

    kernel = np.array([[0, 0], [1, 1], [0, 0]], dtype=np.uint8)
    imgDialsobel = cv2.erode(imgDialsobel, kernel, iterations=2)  # APPLY DILATION

    lines_H = cv2.HoughLinesP(imgDialsobel, 1, np.pi / 2, thres[0], minLineLength=50, maxLineGap=5)
    if (lines_H is not None):
        for line in lines_H:
            x1, y1, x2, y2 = line[0]
            cv2.line(sobelX1, (x1, y1), (x2, y2), (0, 0, 0), 2)
    kernel = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    sobelX1 = cv2.dilate(sobelX1, kernel, iterations=2)  # APPLY DILATION
    cv2.imshow("sobelX111",sobelX1)
    sobelX1[sobelX1 == 0] = 1
    sobelX1[sobelX1 == 255] = 0
    print(sobelX1)
    blankImageforHorizontal = np.zeros((heightImg, widthImg, 3), np.uint8)
    horizontal_projection = np.sum(sobelX1, axis=1) * widthImg / heightImg

    for row in range(heightImg):
        cv2.line(blankImageforHorizontal, (0, row), (int(horizontal_projection[row]), row),(255, 255, 255), 1)
    kernel = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.uint8)
    blankImageforHorizontal = cv2.dilate(blankImageforHorizontal, kernel, iterations=2)  # APPLY DILATION
    cv2.imshow("blankImageforHorizontal",blankImageforHorizontal)


    cv2.line(blankImageforHorizontal, (0,0), (widthImg, 0), (0, 0, 0), 2)
    cv2.line(blankImageforHorizontal, (0,heightImg), (widthImg, heightImg), (0, 0, 0), 2)

    cv2.rectangle(img=blankImageforHorizontal, pt1=(0, heightImg), pt2=( int((widthImg*widthImg*0.95)/heightImg),0 ), color=(0,0,0),thickness=-1 )
    blankImageforHorizontal = cv2.cvtColor(blankImageforHorizontal, cv2.COLOR_BGR2GRAY)
    ret3, imgThresholdBWInvert_h = cv2.threshold(blankImageforHorizontal, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow("imgThresholdBWInvert_h",imgThresholdBWInvert_h)

    contours_h = cv2.findContours(blankImageforHorizontal, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)


    contours_h = contours_h[0] if len(contours_h) == 2 else contours_h[1]

    for cntr in contours_h:
        x, y, w, h = cv2.boundingRect(cntr)
        s = 0
        arr = [x, y, w, h, s]


        # cv2.rectangle(grayCopyForMoodfilling, (x, y), (x + w, y + h), (mood, mood, mood), -1)
        cv2.rectangle(imgCopy2, (x, y), (x + w, y + h), (255,0,0), 3)
    # END OF IMAGE REEMOVING
    #cv2.imshow("imgCopy2", imgCopy2)


    rects = []


    rectsUsed = []

    # Just initialize bounding rects and set all bools to false
    for cnt in contours_h:
        rects.append(cv2.boundingRect(cnt))
    rects.sort(key=getXFromRecty)
    clusters_h = {}
    i = 0

    upperConer = {}
    bottomConer = {}
    imageArray=[]
    a, b = 0, 0
    x1, y1 = 0,0
    x, y, w, h =0,0,widthImg,heightImg
    #contours_h.sort(key=getXFromRecty)
    for cntr in rects:
        x2, y2, w2, h2 = cntr[0],cntr[1],cntr[2],cntr[3]
        clusters_h[i] = [x2, y2, w2, h2]
        if (abs(y2) < 5):
            upperConer = [x2, y2, w2, h2]
            a = 1
        else:
            if (y2 + h2 > heightImg - 5):
                bottomConer = [x2, y2, w2, h2]
                b = 1
            else:

                if(h2+y2>heightImg*0.25):
                    x, y, w, h = x2, y2, w2, h2
                    cropedImg=imgCopy[y1:int(y2+(h2/2)),0:widthImg]
                    x1=x2
                    y1=y2+h2

                    cv2.waitKey(5000)
                    imageArray.append(cropedImg)

        #cv2.waitKey(10)
        i = i + 1
    if(y1>heightImg*0.1):
        cropedImg = imgCopy[y1:heightImg, 0:widthImg]
        imageArray.append(cropedImg)
    if(len(imageArray)==0):
        imageArray.append(img)

    return imageArray

def drow_houghlines(img):
    imgCopy1 = img.copy()
    imgCopy2 = img.copy()
    # edges = cv2.Canny(img, 50, 150, apertureSize=3)
    kernel = np.ones((1, 5))

    # lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    # lap = np.uint8(np.absolute(lap))

    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    sobelCombined = cv2.cvtColor(sobelCombined, cv2.COLOR_BGR2GRAY)

    ret3, imgThresholdBW = cv2.threshold(sobelCombined, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    imgDialsobel = cv2.dilate(imgThresholdBW, kernel, iterations=1)  # APPLY DILATION
    imgThresholdsobel = cv2.erode(imgDialsobel, kernel, iterations=1)  # APPLY EROSION

    cv2.imshow("sobelX",sobelX)
    cv2.imshow("sobelY",sobelY)

    image_rlsa = rlsa.rlsa(image=imgThresholdsobel, horizontal=False, vertical=True, value=12)
    image_rlsa = rlsa.rlsa(image=image_rlsa, horizontal=True, vertical=True, value=5)

    kernel = np.ones((3, 3))

    image_rlsa = cv2.morphologyEx(image_rlsa, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("RLSAa2", image_rlsa)
    kernel = np.ones((3, 3))

    image_rlsa = cv2.morphologyEx(image_rlsa, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("RLSAa", image_rlsa)
    # cv2.imshow("RLSAa1", image_rlsa)
    ret3, image_rlsa = cv2.threshold(image_rlsa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(image_rlsa, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = contours[0] if len(contours) == 2 else contours[1]
    imgCopy=img.copy()
    for cntr in contours:

        x, y, w, h = cv2.boundingRect(cntr)

        if (w * h > heightImg * widthImg / 3000):

            cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)



    #cv2.imshow("result2", imgCopy)

    # print("counters",contours)
    imgOriginal = cv2.cvtColor(imgCopy1, cv2.COLOR_BGR2GRAY)

    mood = utlis.getMostCommonPixel(imgOriginal)

    i = 0
    j = 0

    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        s = 0

        i = i + 1
        z = 4
        if h < 50:
            z = -4
        else:
            z = 4
        if h > 10: cv2.rectangle(imgCopy1, (x + 4, y + 4), (x + w + 2, y + h - z), (mood[0], mood[0], mood[0]), -1)

    img = cv2.cvtColor(imgCopy1, cv2.COLOR_RGB2GRAY)
    # ret3, imgThresholdBWInvert = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)


    kernelHorizontal = np.ones((1, 5))
    kernelVerticle = np.ones((5, 1))

    imgForHorizontalLine = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernelHorizontal)
    imgForVerticalLine = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernelVerticle)

    imgForHorizontalLine_Dilate = cv2.dilate(imgForHorizontalLine, kernelHorizontal, 2)
    imgForVerticalLine_Dilate = cv2.dilate(imgForVerticalLine, kernelVerticle, 2)

    imgForHorizontalLine_Dilate_Close = cv2.morphologyEx(imgForHorizontalLine_Dilate, cv2.MORPH_CLOSE, kernelHorizontal)
    imgForVerticalLine_Dilate_Close = cv2.morphologyEx(imgForVerticalLine_Dilate, cv2.MORPH_CLOSE, kernelVerticle)

    #cv2.imshow('imgForHorizontalLine_Inver_Canny', imgForHorizontalLine_Dilate_Close)
    #cv2.imshow('imgForVerticalLine_Invert_Canny', imgForVerticalLine_Dilate_Close)

    ret3, imgForHorizontalLine_Inver = cv2.threshold(imgForHorizontalLine_Dilate_Close, 0, 255,
                                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret3, imgForVerticalLine_Invert = cv2.threshold(imgForVerticalLine_Dilate_Close, 0, 255,
                                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    imgForHorizontalLine_Inver_Canny = cv2.Canny(imgForHorizontalLine_Inver, 50, 150, apertureSize=3)
    imgForVerticalLine_Invert_Canny = cv2.Canny(imgForVerticalLine_Invert, 50, 150, apertureSize=3)

    lines_V = cv2.HoughLinesP(imgForVerticalLine_Invert_Canny, 1, np.pi, thres[0], minLineLength=50, maxLineGap=50)
    lines_H = cv2.HoughLinesP(imgForHorizontalLine_Inver_Canny, 1, np.pi / 2, thres[0], minLineLength=50, maxLineGap=50)
    print("y",lines_V)
    if lines_V is not None:
        for line1 in lines_V:
            x1, y1, x2, y2 = line1[0]
            for line2 in lines_H:
                x3, y3, x4, y4 = line2[0]

                if x1-x2<10 & ((y1>y3 & y3>y2)|(y1>y4 & y4>y2)|(y3>y1 & y1>y4)|(y3>y2 & y2>y4)):
                    print("have")

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (10, 500)
    # fontScale = 0.3
    # fontColor = (0, 0, 255)
    # lineType = 2
    if (lines_H is not None):
        for line in lines_H:
            x1, y1, x2, y2 = line[0]
            cv2.line(imgCopy2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(imgCopy2,"( "+ str(int(x1))+str(int(y1))+" )( "+str(int(x2))+str(int(y2))+" )",(int((x1+x2)/2),y1),font,fontScale,fontColor,lineType)

    if lines_V is not None:
        for line in lines_V:
            x1, y1, x2, y2 = line[0]
            cv2.line(imgCopy2, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #cv2.imshow("imgCopy1", imgCopy2)
        #cv2.waitKey(500)
    cv2.imshow("bw", imgCopy2)
    return img


while True:
    # success, img = cap.read()
    # cv2.imshow("Video",img)

    cv2.waitKey(1)
    img = cv2.imread("old2.jpg")

    ####################################
    im = img
    row, col = im.shape[:2]
    bottom = im[row - 2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    bordersize = 50
    border = cv2.copyMakeBorder(
        im,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )


    border = cv2.resize(border, (widthImg, heightImg))


    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    thres = utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    ## FIND ALL COUNTOURS
    imgContours = border.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = border.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 1)  # DRAW ALL DETECTED CONTOURS
    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(border, matrix, (widthImg, heightImg))
        # REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored = imgWarpColored[5:imgWarpColored.shape[0] - 10, 5:imgWarpColored.shape[1] - 10]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
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
    # LABELS FOR DISPLAY
    lables = (["Original", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Prespective", "Warp Gray"])
    # stackedImage = utlis.stackImages(imageArray,0.75,lables)
    # cv2.imshow("Result",stackedImage)###########################################all results
    cv2.imshow("imgWarpColored", imgWarpColored)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('imgWarpGray.jpg', imgWarpGray)
        cv2.destroyAllWindows()
        break
utlis.initializeTrackbars()

while True:
    thres = utlis.valTrackbars()
    cv2.waitKey(1)

    gray = cv2.imread("imgWarpGray.jpg")
    grayCopy = gray.copy()
    grayCopyForMoodfilling = gray.copy()
    grayCopy2 = gray.copy()
    grayCopy3 = gray.copy()
    grayCopy4 = gray.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    imgArr=[]
    #drow_houghlines(img)
    ret4, imgThresholdBWInvert = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    imgArr=verticalSeparater(grayCopyForMoodfilling)
    i=0
    for CropedImg in imgArr:

        img = CropedImg.copy()

        CropedImg1=cv2.fastNlMeansDenoisingColored(CropedImg,None,10,10,7,31)

       # CropedImg1 = cv2.GaussianBlur(CropedImg, (7, 7), 0)  # gaussian
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (2,2))
        kernel =  np.ones((0, 0))
        #grayC=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
        imgDial = cv2.dilate(CropedImg1, kernel, iterations=2)  # APPLY DILATION
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
        #imgThreshold = cv2.GaussianBlur(imgThreshold, (9, 9), 1)  # gaussian
        out_gray = cv2.divide(CropedImg, imgThreshold, scale=255)

        ret3, bw1 = cv2.threshold(cv2.cvtColor(imgThreshold, cv2.COLOR_RGB2GRAY), ret4-10, 255, cv2.THRESH_BINARY_INV)
        contours = cv2.findContours(bw1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mood = utlis.getMostCommonPixel(cv2.cvtColor(CropedImg,cv2.COLOR_RGB2GRAY))


        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            epsilon = 0.01 * cv2.arcLength(cntr, True)
            approx = cv2.approxPolyDP(cntr, epsilon, True)
            hull = cv2.convexHull(cntr)


            if((x==0 & y==0) | (x==0 & y+h==CropedImg.shape[0]) | (x==CropedImg.shape[1] & y==0 ) | (x==CropedImg.shape[1] & y==CropedImg.shape[0]) ) | (h*w<widthImg*heightImg*0.005 ):
                continue
            #cv2.rectangle(img, (x, y), (x + w, y + h), (mood[0], mood[0], mood[0]), -1)
            #cv2.drawContours(img, [cntr], -1, (0, 255, 0), 2)
            #cv2.drawContours(img, [approx], -1, (255, 255, 0), 2)
            img2=cv2.drawContours(img, [hull], -1, (mood[0], mood[0], mood[0]),thickness= -1)
            img2=cv2.drawContours(img, [hull], -1, (mood[0], mood[0], mood[0]),thickness= 4)

        # cv2.imshow(str(i),CropedImg)
        # cv2.imshow(str(i)+"d",CropedImg1)
        # cv2.imshow(str(i)+"t",imgThreshold)
        cv2.imshow(str(i)+"NN",img)
        imgWithLine = drow_houghlines(img)
        # #cv2.imshow(str(i)+"C",sobelCombined)
        # cv2.imshow(str(i)+"G",bw1)
        #drow_houghlines(img)

        cv2.waitKey(1)
        i=i+1




    gapSize=pixeldensity(gray)
    # image removing start
    # cv2.imshow("bGaus", gray)

    # cv2.imshow("aGaus", gray)


    # imgThresholdBW = cv2.adaptiveThreshold(imgThreshold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 10)

    # cv2.imshow("imgThresholdBW", imgThresholdBWInvert)
    # (thresh, imgThresholdBWInvert) = cv2.threshold(imgThresholdBW, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # contours = cv2.findContours(imgThresholdBWInvert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    #
    #
    # mood = utlis.getMostCommonPixel(gray)
    # print("mood", mood)
    # i = 0
    # j = 0
    #
    # for cntr in contours:
    #     x, y, w, h = cv2.boundingRect(cntr)
    #     s = 0
    #     arr = [x, y, w, h, s]
    #
    #     i = i + 1
    #     # cv2.rectangle(grayCopyForMoodfilling, (x, y), (x + w, y + h), (mood, mood, mood), -1)
    #     cv2.rectangle(grayCopyForMoodfilling, (x, y), (x + w, y + h), (mood[0], mood[0], mood[0]), -1)
    # # END OF IMAGE REEMOVING
    # # cv2.imshow("grayCopyForMoodfilling", grayCopyForMoodfilling)



    # cv2.imshow("imgWithLine", imgWithLine)

    (thresh, blackAndWhiteInvertImage) = cv2.threshold(cv2.cvtColor(imgThreshold, cv2.COLOR_BGR2GRAY), 0, 255,
                                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # THRESHOLD "Invert b&w image"
    # blackAndWhiteInvertImage11 = cv2.adaptiveThreshold(imgThreshold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
    # blackAndWhiteInvertImage22 = cv2.adaptiveThreshold(imgThreshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
    # cv2.imshow("blackAndWhiteInvertImage11", blackAndWhiteInvertImage11)
    # cv2.imshow("blackAndWhiteInvertImage", blackAndWhiteInvertImage)
    # cv2.imshow("blackAndWhiteInvertImage22", blackAndWhiteInvertImage22)

    # (thresh, bw1) = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #ret3, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 10)

    # ret3, bw1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # ret3, bw2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY )
    # cv2.imshow("dailHori", imgDial)

    # cv2.imshow("bwFirst", bw)
    # cv2.imshow("bwFirst1", bw1)
    # cv2.imshow("bwFirst2", bw2)
    # blackAndWhiteInvertImage = cv2.dilate(blackAndWhiteInvertImage, kernel, iterations=2)
    # fainalBWImage = blackAndWhiteInvertImage + bw
    # # cv2.imshow("fainal", fainalBWImage)
    #
    # # RLSA
    # # image_rlsa_hori = rlsa.rlsa(image=fainalBWImage,horizontal=True,vertical=False,value=thres[2])
    # # image_rlsa_verti = rlsa.rlsa(image=fainalBWImage,horizontal=False,vertical=True,value=thres[2])
    # image_rlsa = rlsa.rlsa(image=fainalBWImage, horizontal=True, vertical=True, value=5)
    # # cv2.imshow("RLSAa", image_rlsa)
    # kernel = np.ones((5, 1))
    #
    # image_rlsa = cv2.morphologyEx(image_rlsa, cv2.MORPH_CLOSE, kernel)
    # # cv2.imshow("RLSAa2", image_rlsa)
    # kernel = np.ones((1, 5))
    #
    # image_rlsa = cv2.morphologyEx(image_rlsa, cv2.MORPH_CLOSE, kernel)
    # # cv2.imshow("RLSAa1", image_rlsa)
    #
    # result = img.copy()
    # # image_rlsa = abs(image_rlsa - 255)
    # (thresh, image_rlsa) = cv2.threshold(image_rlsa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #

    # Merge threshold for x coordinate distance
    # cv2.imshow("second", bw3)


# cv2.imshow("bw", bw)
# cv2.imshow("imgDial", imgDial)
# cv2.imshow("imgThreshold", imgThreshold)
# cv2.imshow("blackAndWhiteImage", blackAndWhiteImage)




########################################################################
# import cv2
# import numpy as np
#
# image = cv2.imread('images/example.png')
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
# kernel = np.ones((5, 5), np.uint8)
# img_dilated = cv2.dilate(thresh, kernel, iterations = 1)
#
# cnts, _ = cv2.findContours(img_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#
# cv2.imwrite("images/result.png", image)
cv2.waitKey(0)

