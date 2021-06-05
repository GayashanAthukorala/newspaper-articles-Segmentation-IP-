import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import cv2
from pythonRLSA import rlsa
imghight=640
imgwidth=480
img = cv2.imread('vidu 2_1.jpg')
img = cv2.resize(img, (imgwidth, imghight))



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



#2021/2/19
gray = cv2.resize(gray, (imgwidth, imghight))
kernel = np.ones((3,3))
imgDial = cv2.dilate(gray, kernel, iterations=2) # APPLY DILATION
imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
(thresh, blackAndWhiteImage) = cv2.threshold(imgThreshold, 170, 255, cv2.THRESH_BINARY_INV)
(thresh, bw) = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
fainal=blackAndWhiteImage+bw

result = img.copy()
#image_rlsa = abs(image_rlsa - 255)
contours = cv2.findContours(blackAndWhiteImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:

    x, y, w, h = cv2.boundingRect(cntr)
    if(w*h>imghight*imgwidth/3000):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print("x,y,w,h:", x, y, w, h)



cv2.imshow("imgDial", imgDial)
cv2.imshow("imgThreshold", imgThreshold)
cv2.imshow("blackAndWhiteImage", blackAndWhiteImage)
cv2.imshow("fainal", fainal)
cv2.imshow("result1", result)

#/2/19
# conver to binnary
(thresh, blackAndWhiteImage) = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
cv2.imwrite('binaryImg.jpg', blackAndWhiteImage)

# RLSA
image_rlsa = rlsa.rlsa(blackAndWhiteImage, 1, 1, 7)
cv2.imwrite('RLSA_img.jpg', image_rlsa)
cv2.imshow("RLSAb", image_rlsa)
kernel = np.ones((7,1))
image_rlsa=cv2.morphologyEx(image_rlsa,cv2.MORPH_CLOSE,kernel)
cv2.imshow("RLSAa1", image_rlsa)
kernel = np.ones((1,7))
image_rlsa=cv2.morphologyEx(image_rlsa,cv2.MORPH_CLOSE,kernel)
cv2.imshow("RLSAa2", image_rlsa)
# get contours

result = img.copy()
image_rlsa = abs(image_rlsa - 255)
cv2.imshow("rlsaBC",image_rlsa)
contours = cv2.findContours(image_rlsa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    x, y, w, h = cv2.boundingRect(cntr)
    if ((w * h > imghight * imgwidth / 3000)& (w>imgwidth/20)&(h>imghight/20)):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #print("x,y,w,h:", x, y, w, h)

# save resulting image
cv2.imwrite('result.jpg', result)

# show thresh and result
cv2.imshow("bounding_box", result)
cv2.waitKey(0)
