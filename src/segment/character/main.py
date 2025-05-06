import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
from segment import mask_bounding_boxes, crop_to_content

if __name__ == "__main__":
    image_paths = os.listdir("line-crops")

    for image_path in image_paths:
        image_names = os.listdir("line-crops/" + image_path)

        for image_name in image_names:
            path = "line-crops/" + image_path + "/" + image_name
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8, ltype=cv2.CV_32S)

            for i in range(1, num_labels):

                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                (cx, cy) = centroids[i]
                
                if area < 50:
                    continue

                mask = mask_bounding_boxes(image, stats, [i], padding=2)
                img = crop_to_content(mask, padding=2)

                os.makedirs('char-crops/' + image_path + "/" + image_name , exist_ok=True)
                cv2.imwrite(f"char-crops/{image_path}/{image_name}/char_{i}.png", img)       