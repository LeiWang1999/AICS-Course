from PIL import Image
import numpy as np
image = Image.open('hare.jpg') # 图片是400x300 宽x高
print(type(image)) # out: PIL.JpegImagePlugin.JpegImageFile
print(image.size)  # out: (400,300)
print(image.mode) # out: 'RGB'
print(image.getpixel((0,0))) # out: (143, 198, 201)
image = np.array(image,dtype=np.float32) # image = np.array(image)默认是uint8
print(image.shape) # out: (100, 200, 3)
# 神奇的事情发生了，w和h换了，变成(h,w,c)了
# 注意ndarray中是 行row x 列col x 维度dim 所以行数是高，列数是宽