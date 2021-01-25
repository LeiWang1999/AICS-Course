import cv2
import numpy as np
image = cv2.imread('hare.jpg')
print(type(image)) # out: numpy.ndarray
print(image.dtype) # out: dtype('uint8')
print(image.shape) # out: (300, 400, 3) (h,w,c) 和skimage类似
print(image)  # BGR
