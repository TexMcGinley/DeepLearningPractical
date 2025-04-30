import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN

image_paths = os.listdir("image-data")

for image_path in image_paths:
    image_path = "image-data/" + image_path

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1024, 1024))
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.erode(image, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4, ltype=cv2.CV_32S)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    y_coords = []
    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cx, cy) = centroids[i]

        
        
        if area < 25:
            continue

        y_coords.append(cy)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.circle(image, (int(cx), int(cy)), 2, (0, 0, 255), -1)  

    plt.scatter(range(len(y_coords)), y_coords)
    plt.show()
    cv2.imshow("Image", image)
    cv2.waitKey(0)