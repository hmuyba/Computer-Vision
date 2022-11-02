# Importamos las librerías necesarias. En este caso, las dos más importantes son Numpy y OpenCV.
import argparse

import cv2
import numpy as np

# Los argumentos de entrada definen la ruta a la imagen que será segmentada, así como el número de *clusters* o grupos
# a hallar mediante la aplicación de K-Means.
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True, type=str, help='Ruta a la imagen a segmentar.')
argument_parser.add_argument('-k', '--num-clusters', default=3, type=int,
                             help='Número de clusters para K-Means (por defecto = 3).')
arguments = vars(argument_parser.parse_args())

# Cargamos la imagen de entrada.
image = cv2.imread(arguments['image'])

# Creamos una copia para poderla manipular a nuestro antojo.
image_copy = np.copy(image)

# Mostramos la imagen y esperamos que el usuario presione cualquier tecla para continuar.
cv2.imshow('Imagen', image)
cv2.waitKey(0)

# Convertiremos la imagen en un arreglo de ternas, las cuales representan el valor de cada pixel. En pocas palabras,
# estamos aplanando la imagen, volviéndola un vector de puntos en un espacio 3D.
pixel_values = image_copy.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Abajo estamos aplicando K-Means. Como siempre, OpenCV es un poco complicado en su sintaxis, así que vamos por partes.

# Definimos el criterio de terminación del algoritmo. En este caso, terminaremos cuando la última actualización de los
# centroides sea menor a *epsilon* (cv2.TERM_CRITERIA_EPS), donde epsilon es 1.0 (último elemento de la tupla), o bien
# cuando se hayan completado 10 iteraciones (segundo elemento de la tupla, criterio cv2.TERM_CRITERIA_MAX_ITER).
stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Este es el número de veces que se correrá K-Means con diferentes inicializaciones. La función retornará los mejores
# resultados.
number_of_attempts = 10

# Esta es la estrategia para inicializar los centroides. En este caso, optamos por inicialización aleatoria.
centroid_initialization_strategy = cv2.KMEANS_RANDOM_CENTERS

# Ejecutamos K-Means con los siguientes parámetros:
# - El arreglo de píxeles.
# - K o el número de clusters a hallar.
# - None indicando que no pasaremos un arreglo opcional de las mejores etiquetas.
# - Condición de parada.
# - Número de ejecuciones.
# - Estrategia de inicialización.
#
# El algoritmo retorna las siguientes salidas:
# - Un arreglo con la distancia de cada punto a su centroide. Aquí lo ignoramos.
# - Arreglo de etiquetas.
# - Arreglo de centroides.
_, labels, centers = cv2.kmeans(pixel_values,
                                arguments['num_clusters'],
                                None,
                                stop_criteria,
                                number_of_attempts,
                                centroid_initialization_strategy)

# Aplicamos las etiquetas a los centroides para segmentar los pixeles en su grupo correspondiente.
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# Debemos reestructurar el arreglo de datos segmentados con las dimensiones de la imagen original.
segmented_image = segmented_data.reshape(image_copy.shape)

# Mostramos la imagen segmentada resultante.
cv2.imshow('Imagen segmentada', segmented_image)
cv2.waitKey(0)
