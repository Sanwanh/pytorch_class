# import torch
# import torchvision


# print(torch.__version__)
# print(torchvision.__version__)

# import numpy as np

# a=np.arange(10)
# print(a)

import cv2

img = cv2.imread('3sjM3t9.jpg')
img = cv2.resize(img,(400,400))
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()