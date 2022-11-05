
import cv2
import numpy as np
from matplotlib import pyplot as plt


def umbralizacion_Demostrativa():

    img = cv2.imread('escala.jpg', 0)

    u, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    u, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    u, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    u, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    u, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    images = [img, th1, th2, th3, th4, th5]
    labels = ['Original', 'Binary', 'Binary Inv', 'Trunc', 'To zero', 'To zero Inv']

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.imshow(images[i], 'gray', vmin=0, vmax=255);
        plt.title(labels[i])
        plt.xticks([]), plt.yticks([])

    plt.show()
    print(u)


def umbralizacion_Adaptativa():

    img = cv2.imread('caballo.jpg', 0)
    th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    u, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    images = [th1, th2]
    labels = ['ADAPTIVE_THRESH_MEAN_C', 'SIMPLE']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(labels[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def umbralizacion_Automatica():

    img1 = cv2.imread('gato.jpg', 0)
    img2 = cv2.imread('banana.jpg', 0)
    img3 = cv2.imread('water_coins.jpg', 0)

    ret1, th1 = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, th2 = cv2.threshold(img2, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret3, th3 = cv2.threshold(img3, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print('Umbral de th1:', ret1)
    print('Umbral de th2:', ret2)
    print('Umbral de th3:', ret3)

    images = [img1, th1, img2, th2, img3, th3]
    titles = ['gato', 'OTSU', 'banana', 'OTSU', 'monedas', 'OTSU']

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


umbralizacion_Demostrativa()
umbralizacion_Adaptativa()
umbralizacion_Automatica()