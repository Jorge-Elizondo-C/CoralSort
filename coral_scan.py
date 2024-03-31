import cv2
import sys  # to access the system
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from matplotlib import image as mpimg

plt.title("Coral")

# image = mpimg.imread("C:/Users/Jorge/Downloads/drive-download-20240329T071726Z-001/DSCN2178.jpg")
# plt.imshow(image)
# # plt.show()

img = cv2.imread("C:/Users/Jorge/Downloads/drive-download-20240329T071726Z-001/DSCN2178.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
# plt.show()

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
plt.imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB))
# plt.show()
edged = cv2.Canny(bfilter, 20, 30) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.show()
#
# keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(keypoints)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
#
# location = None
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 10, True)
#     if len(approx) == 4:
#         location = approx
#         break
#
# location
# mask = np.zeros(gray.shape, np.uint8)
# new_image = cv2.drawContours(mask, [location], 0, 255, -1)
# new_image = cv2.bitwise_and(img, img, mask=mask)
#
# plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
# plt.show()
#
# (x,y) = np.where(mask==255)
# (x1, y1) = (np.min(x), np.min(y))
# (x2, y2) = (np.max(x), np.max(y))
# cropped_image = gray[x1:x2+1, y1:y2+1]
#
# plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
# plt.show()
reader = easyocr.Reader(['en'])
result = reader.readtext(edged)
print(result)
