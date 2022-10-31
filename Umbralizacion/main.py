
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('water_coins.jpg', 1)
# cv2.imshow('Foto original', img)
# cv2.waitKey(0)

u, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
u, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
u, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
u, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
u, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

images = [img, th1, th2, th3, th4, th5]
labels = ['Original','Binary', 'Binary Inv', 'Trunc', 'To zero', 'To zero Inv']

for i in range(6):
    plt.subplot(3,2,i+1)
    plt.imshow(images[i],'gray',vmin=0,vmax=255);
    plt.title(labels[i])
    plt.xticks([]), plt.yticks([])

plt.show()
print(u)
