#contours,_= cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print('x='.format(len(contours)))



   # print('x=',box)
   # img = cv2.drawContours(img,[1],0,(0,254,0),thickness=3)
#cv2.imshow('box',img)

#///////////////
#edges = cv2.Canny(image_rlsa_horizontal,50,150,apertureSize = 3)

##cv2.imshow('edges',edges)

#minLineLength = 100
#maxLineGap = 10
#lines = cv2.HoughLinesP(image_rlsa_horizontal,1,np.pi/360,50,np.array([]),minLineLength,maxLineGap)

#for line in lines:
 #   for x1,y1,x2,y2 in line:
 #       cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
##cv2.imwrite('houghlines5.jpg',img)
#cv2.imshow('houghlines',img)

# #if(i>0):
        # print("ddddddddddd",i)
        # if(boundryArry[int(i)][0]==boundryArry[int(i)][0]):
        #
        #     print("test1111", boundryArry[0][0])
        #
        # i = i + 1
        #         if(abs(boundryArry[str(i)][3]-boundryArry[str(i-1)][3])<3 & abs(boundryArry[str(i-1)][1]-boundryArry[str(i)][1]-boundryArry[str(i)][2])<35):
        #             newboundryarry[j]=[boundryArry[str(i-1)][1],boundryArry[str(i-1)][0],boundryArry[str(i-1)][2]+boundryArry[str(i)][2]+abs(boundryArry[str(i-1)][1]-boundryArry[str(i)][1]-boundryArry[str(i)][2])<35,max(boundryArry[str(i)][3],boundryArry[str(i-1)][3])]
        #         else:
        #             j=j+1
        #             newboundryarry[j] = arr
        # else:
        #     newboundryarry[j] = arr
        #     j=j+1
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
    return
####################################################
  newboundryarry = {}
    for i in range(0, len(boundryArry)):
        x1, y1, w1, h1, s1 = boundryArry[i][0], boundryArry[i][1], boundryArry[i][2], boundryArry[i][3], boundryArry[i][
            4]
        if (s1 == 1): continue
        if (w1 * h1 < heightImg * widthImg / 3000): continue

        for j in range(i + 1, len(boundryArry)):
            x2, y2, w2, h2, s2 = boundryArry[j][0], boundryArry[j][1], boundryArry[j][2], boundryArry[j][3], \
                                 boundryArry[j][4]
            if (w2 * h2 < heightImg * widthImg / 3000): continue

            c1 = random.randint(0, 255)
            c2 = random.randint(0, 255)
            c3 = random.randint(0, 255)

            if (x1 >= x2 + w2) or (x1 + w1 <= x2) or (y1 + h1 <= y2) or (y1 >= y2 + h2):
                continue
            else:
                # cv2.rectangle(grayCopy2, (x1, y1), (x1 + w1, y1 + h1), (c1,c2,c3), 2)
                # cv2.rectangle(grayCopy2, (x2, y2), (x2 + w2, y2 + h2), (c1,c2,c3), 2)

                x3, y3 = min(x1, x2), y2
                if ((x1 > x2) & (x2 + w2 < x1 + w1)):
                    w3 = x1 - x2 + w1
                if ((x1 > x2) & (x2 + w2 > x1 + w1)):
                    w3 = w2
                if ((x1 < x2) & (x2 + w2 > x1 + w1)):
                    w3 = x2 - x1 + w2
                if ((x1 < x2) & (x2 + w2 < x1 + w1)):
                    w3 = w1
                if (y2 < y1):
                    h3 = y1 - y2 + h1
                if (y2 + h2 > y1 + h1):
                    h3 = h2
                # cv2.imshow("result3", grayCopy2)

                # cv2.waitKey(1000)
                print("seted")
                #boundryArry[j][4] = 1
                # cv2.rectangle(grayCopy2, (x3, y3), (x3 + w3, y3 + h3), (c1, c2, c3), 2)
                x1, y1, w1, h1 = x3, y3, w3, h3
            #####################
        #newboundryarry[z] = [x1, y1, w1, h1, 0]
        # print("ADED",x3, y3, w3, h3)
        z = z + 1

        # x1, y1, w1, h1,s1 = min(x1,x2), y2, max(x1,x2)-min(x2,x1)-w1 if min(x2,x1)==x2 else w2, max(y1,y2)-min(y1,y2)- h2 if min(y1,y2)==y2 else h1,1

        #

    # print("counter:", boundryArry)
    # print("counter2:", newboundryarry)

    for cntr in boundryArry:
        x, y, w, h =  cv2.boundingRect(cntr)
        if (w * h > heightImg * widthImg / 3000):
            if (h < heightImg / 30):
                # cv2.rectangle(bw, (x, y), (x + w, y + h), (255, 255, 255), -1)
                cv2.rectangle(grayCopy2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(grayCopy2, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.waitKey(1000)
            # cv2.imshow("result3", grayCopy2)

    newboundryarry = merge_taitels(newboundryarry)
    for i in range(0, len(newboundryarry)):
        x, y, w, h = newboundryarry[i][0], newboundryarry[i][1], newboundryarry[i][2], newboundryarry[i][3]
        if (w * h > heightImg * widthImg / 3000):
            if (h < heightImg / 30):
                # cv2.rectangle(bw, (x, y), (x + w, y + h), (255, 255, 255), -1)
                cv2.rectangle(grayCopy3, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(grayCopy3, (x, y), (x + w, y + h), (0, 0, 255), 2)
# x3, y3 = min(x1, x2), y2
                # if ((x1 > x2) & (x2 + w2 < x1 + w1)):
                #     w3 = x1 - x2 + w1
                # if ((x1 > x2) & (x2 + w2 > x1 + w1)):
                #     w3 = w2
                # if ((x1 < x2) & (x2 + w2 > x1 + w1)):
                #     w3 = x2 - x1 + w2
                # if ((x1 < x2) & (x2 + w2 < x1 + w1)):
                #     w3 = w1
                # if (y2 < y1):
                #     h3 = y1 - y2 + h1
                # if (y2 + h2 > y1 + h1):
                #     h3 = h2
                # cv2.imshow("result3", grayCopy2)
                ##########################################################################################################################################\
contours = cv2.findContours(image_rlsa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = contours[0] if len(contours) == 2 else contours[1]
mood = utlis.getMostCommonPixel(gray)
# print("mood2",mood)
# boundryArry = {}

# i = 0
# j = 0

##########################

##########################

# contours.sort(key=getXFromRectx)
# for cntr in contours:
#
#     x, y, w, h = cv2.boundingRect(cntr)
#     s = 0
#     arr = [x, y, w, h, s]
#     boundryArry[i] = arr
#     i = i + 1
#     if (w * h > heightImg * widthImg / 3000):
#         if (h < heightImg / 32):
#             # cv2.rectangle(bw, (x, y), (x + w, y + h), (255, 255, 255), -1)
#             cv2.rectangle(grayCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         else:
#             cv2.rectangle(grayCopy, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         # print("counter:", arr)
#
#         # cv2.imshow("result2", grayCopy)
#
#         # cv2.waitKey(1000)
#
# # print("counter2:", newboundryarry)
# z = 0
# # #boundryArry.sort(key = getXFromRect)
# newboundryarry = {}

# x1, y1, w1, h1,s1 = min(x1,x2), y2, max(x1,x2)-min(x2,x1)-w1 if min(x2,x1)==x2 else w2, max(y1,y2)-min(y1,y2)- h2 if min(y1,y2)==y2 else h1,1
# #     #
# #
# # print("counter:", boundryArry)
# # print("counter2:", newboundryarry)
#
# for i in range(0, len(newboundryarry)):
#     x, y, w, h = newboundryarry[i][0], newboundryarry[i][1], newboundryarry[i][2], newboundryarry[i][3]
#     if (w * h > heightImg * widthImg / 3000):
#         if (h < heightImg / 30):
#              # cv2.rectangle(bw, (x, y), (x + w, y + h), (255, 255, 255), -1)
#             cv2.rectangle(grayCopy2, (x, y), (x + w, y + h), (c2, c1, c2), 2)
#         else:
#             cv2.rectangle(grayCopy2, (x, y), (x + w, y + h), (c3, c3, c2), 2)
#         cv2.waitKey(1000)
#         cv2.imshow("result3", grayCopy2)
# #
# # newboundryarry = merge_taitels(newboundryarry)
# # for i in range(0, len(newboundryarry)):
# #     x, y, w, h = newboundryarry[i][0], newboundryarry[i][1], newboundryarry[i][2], newboundryarry[i][3]
# #     if (w * h > heightImg * widthImg / 3000):
# #         if (h < heightImg / 30):
# #             # cv2.rectangle(bw, (x, y), (x + w, y + h), (255, 255, 255), -1)
# #             cv2.rectangle(grayCopy3, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #         else:
# #             cv2.rectangle(grayCopy3, (x, y), (x + w, y + h), (0, 0, 255), 2)
# #
# cv2.imshow("image_rlsa5",image_rlsa)

newboundryarry, _ = cv2.findContours(image_rlsa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Array of initial bounding rects
rects = []
rectsAfterMerg = []
# already been used
rectsUsed = []

# Just initialize bounding rects and set all bools to false
for cnt in newboundryarry:
    rects.append(cv2.boundingRect(cnt))
    rectsUsed.append(False)
boundryArry = {}

rects.sort(key=getXFromRectx)
imgS1 = grayCopy.copy()
# for cntr in rects:
#
#     x, y, w, h = cntr
#     s = 0
#     arr = [x, y, w, h, s]
#     boundryArry[i] = arr
#
#     i = i + 1
#     if (w * h > heightImg * widthImg / 3000):
#         if (h < heightImg / 32):
#             # cv2.rectangle(bw, (x, y), (x + w, y + h), (255, 255, 255), -1)
#             cv2.rectangle(imgS1, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         else:
#             cv2.rectangle(imgS1, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         # print("counter:", arr)
#     cv2.imshow("imgS1", imgS1)
#     #cv2.waitKey(100)
grayCopy5 = grayCopy2.copy()

###############################################################################################################################
for i in range(0, len(rects)):
    x1, y1, w1, h1 = rects[i][0], rects[i][1], rects[i][2], rects[i][3]
    # cv2.rectangle(grayCopy2, (x1, y1), (x1 + w1, y1 + h1), (255,0,50), 2)

    if (rectsUsed[i]): continue

    # if (w1 * h1 < heightImg * widthImg / 3000): continue

    for j in range(i + 1, len(rects)):
        x2, y2, w2, h2 = rects[j][0], rects[j][1], rects[j][2], rects[j][3]

        if (x2 > x1 + w1): break

        # c1 = random.randint(0, 255)
        # c2 = random.randint(0, 255)
        # c3 = random.randint(0, 255)

        # if (x1 >= x2 + w2) or (x1 + w1 <= x2) or (y1 + h1 <= y2) or (y1 >= y2 + h2):
        if ((x1 <= x2) and (x2 <= x1 + w1)) and (
                (y1 <= y2 and (y2 <= y1 + h1)) or ((y1 <= y2 + h2) and (y2 + h2 <= y1 + h1)) or (
                y2 <= y1 and (y1 + h1 <= y2 + h2))):
            # cv2.rectangle(grayCopy2, (x1, y1), (x1 + w1, y1 + h1), (c1, c2, c3), 2)
            # cv2.rectangle(grayCopy2, (x2, y2), (x2 + w2, y2 + h2), (c1, c2, c3), 2)
            x1, y1, w1, h1 = mergCounters([x1, y1, w1, h1], [x2, y2, w2, h2])
            # cv2.rectangle(grayCopy2, (x1, y1), (x1 + w1, y1 + h1), (0, 0, c3), 2)

            # rectsi[0], rectsi[1], rectsi[2], rectsi[3]=x1,y1,w1,h1
            # rectsj[0], rectsj[1], rectsj[2], rectsj[3]=x1,y1,w1,h1
            rectsUsed[j] = True
            # cv2.imshow("result3", grayCopy2)

    rectsAfterMerg.append((x1, y1, w1, h1))
imgtest = gray.copy()
for cntr in rectsAfterMerg:

    x, y, w, h = cntr
    s = 0
    arr = [x, y, w, h, s]
    boundryArry[i] = arr
    i = i + 1

    if (w * h > heightImg * widthImg / 3000):
        if (h < heightImg / 32):
            # cv2.rectangle(bw, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(imgtest, (x, y), (x + w, y + h), (0, 255, 100), 2)
        else:
            cv2.rectangle(imgtest, (x, y), (x + w, y + h), (100, 0, 255), 2)
        # print("counter:", arr)
    cv2.imshow("imgtest", imgtest)

# merg titles
rectForMergTitle = []
rectAfterMergTitle = []
rectsUsed = []
for cnt in rectsAfterMerg:
    rectForMergTitle.append(cnt)
    rectsUsed.append(False)
rectForMergTitle.sort(key=getXFromRectx)
meanHight = 0
for i in range(0, len(rectForMergTitle)):
    x1, y1, w1, h1 = rectForMergTitle[i][0], rectForMergTitle[i][1], rectForMergTitle[i][2], rectForMergTitle[i][3]
    # cv2.rectangle(grayCopy5, (x1, y1), (x1 + w1, y1 + h1), (255,0,50), 2)
    minHight = h1
    # print("counter:", (x1, y1, w1, h1))

    if (rectsUsed[i]): continue

    for j in range(i + 1, len(rectForMergTitle)):
        x2, y2, w2, h2 = rectForMergTitle[j][0], rectForMergTitle[j][1], rectForMergTitle[j][2], rectForMergTitle[j][3]

        c1 = random.randint(0, 255)
        c2 = random.randint(0, 255)
        c3 = random.randint(0, 255)

        if (((abs(h1 - h2) <= minHight * 0.4) or ((abs(x2 - x1 - w1) < 15) and (abs(h1 - h2) <= minHight * 0.6))) and (
                abs(y1 - y2) < 15) and (abs(x2 - x1 - w1) < gapSize * 0.8) and (minHight < heightImg / 32)):
            cv2.rectangle(grayCopy5, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
            cv2.rectangle(grayCopy5, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
            cv2.line(grayCopy5, (14, 81), (150, 88), (255, 0, 200), 2)
            x1, y1, w1, h1 = mergCounters([x1, y1, w1, h1], [x2, y2, w2, h2])
            cv2.rectangle(grayCopy5, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
            if (minHight > h1):
                minHight = h1
            rectsUsed[j] = True
        # cv2.imshow("grayCopy5", grayCopy5)

    rectAfterMergTitle.append((x1, y1, w1, h1))

imgtest22 = gray.copy()
for cntr in rectAfterMergTitle:

    x, y, w, h = cntr
    s = 0
    arr = [x, y, w, h, s]
    boundryArry[i] = arr
    i = i + 1

    if (w * h > heightImg * widthImg / 3000):
        if (h < heightImg / 32):
            # cv2.rectangle(bw, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(imgtest22, (x, y), (x + w, y + h), (0, 255, 100), 2)
        else:
            cv2.rectangle(imgtest22, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.waitKey(5000)
    cv2.line(imgtest22, (14, 81), (150, 81), (20, 70, 60), 2)
    # cv2.imshow("imgtest22", imgtest22)
###############################################################################################################################
# Array of accepted rects
acceptedRects = []


############################################################################################################################################
norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)