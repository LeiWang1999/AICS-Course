import skimage
from skimage import io,transform
import numpy as np
image= io.imread('hare.jpg')
# 第一个参数是文件名可以是网络地址，第二个参数默认为False，True时为灰度图
print(type(image)) # out: numpy.ndarray
print(image.dtype) # out: dtype('uint8')
print(image.shape)  # out: (300, 400, 3) (h,w,c)前面介绍了ndarray的特点
print(image)
# mode也是RGB
'''
注意此时image里都是整数uint8,范围[0-255]
array([
        [ [143, 198, 201 (dim=3)],[143, 198, 201],... (w=200)],
        [ [143, 198, 201],[143, 198, 201],... ],
        ...(h=100)
      ], dtype=uint8)

'''
'''
此时image范围变为[0-1]
array([[ 0.73148549,  0.73148549,  0.73148549, ...,  0.73148549,
         0.73148549,  0.73148549],
       [ 0.73148549,  0.73148549,  0.73148549, ...,  0.73148549,
       .....]])
'''
print(image.dtype) # out: dtype('float64')
print(skimage.img_as_float(image))  # out: float64
# img_as_float可以把image转为double，即float64