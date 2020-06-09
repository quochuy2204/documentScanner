#This project simulates the process of processing scanned images with opencv library in python
#import image
import cv2
input_image  = 'mydoc.jpg'
image = cv2.imread(input_image)
cv2.imshow('Input', image)
cv2.waitKey()

#Remove noise and find the edges of the text in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray,(3,3))
edge = cv2.Canny(blur, 50, 300, 3)

cv2.imshow("gray", gray)
cv2.imshow("blur", blur)
cv2.imshow("edge", edge)
cv2.waitKey()

#Find contours, Arrange the contour by descending area, set the biggest contour to be the document edge
cnts = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
import imutils
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
cnts = cnts[:1]

p = cv2.arcLength(cnts[0], True)
r = cv2.approxPolyDP(cnts[0], 0.02*p, True)
cv2.drawContours(image, [r], -1, (0,0,255), 3)
cv2.imshow("Draw", image)
cv2.waitKey()

#Reshape our ROI for (4,2) - 4 coordinates, each with x, y
r = r.reshape(4,2)
import numpy as np
rect = np.zeros((4,2), dtype='float32')

#sum the coordinates by column
s = np.sum(r, axis=1)
rect[0] = r[np.argmin(s)]
rect[2] = r[np.argmax(s)]

#Calculate the difference between coordinates in columns
diff = np.diff(r, axis=1)
rect[1] = r[np.argmin(diff)]
rect[3] = r[np.argmax(diff)]

#Calculate the width and height of the document
(tl, tr, br, bl) = rect
width1 = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
width2 = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
Width = max(int(width1), int(width2))
height1 = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
height2 = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
Height = max(int(height1), int(height2))

#New coordinates of the text
new_rect = np.array([
    [0,0],
    [Width-1, 0],
    [Width-1, Height-1],
    [0, Height-1]], dtype="float32")

#rotate and crop
M = cv2.getPerspectiveTransform(rect, new_rect)
output = cv2.warpPerspective(image, M, (Width, Height))
cv2.imshow("Output",output)
cv2.waitKey()

#use threshold
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
_, output_final = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
cv2.imshow("Ouput", output_final)
cv2.waitKey()