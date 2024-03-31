import cv2
import sys  # to access the system
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from matplotlib import image as mpimg

image = cv2.imread('C:/Users/Jorge/Downloads/drive-download-20240329T071726Z-001/DSCN2178.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.show()

orig_image = image.copy()
cv2.imshow('original image', orig_image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(orig_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('Bounding rect', orig_image)
cv2.waitKey(0)

#calculate accuracy as a percent of contour perimeter
accuracy = 0.03*cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, accuracy, True)
cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
cv2.imshow('Approx polyDP', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
