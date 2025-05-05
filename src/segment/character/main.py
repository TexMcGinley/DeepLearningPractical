import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d

if __name__ == "__main__":
    image_paths = os.listdir("image_crops")

    for image_path in image_paths:
        image_names = os.listdir("image_crops/" + image_path)

        for image_name in image_names:
            path = "image_crops/" + image_path + "/" + image_name
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8, ltype=cv2.CV_32S)

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for i in range(1, num_labels):

                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                (cx, cy) = centroids[i]

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.circle(image, (int(cx), int(cy)), 2, (0, 0, 255), -1)

            cv2.imshow("image", image)
            cv2.waitKey(0)         