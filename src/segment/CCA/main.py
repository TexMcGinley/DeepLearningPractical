import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d

image_paths = os.listdir("image-data")

def line_projection_histogram(image, show_plot=True):
    binary = (image == 255).astype(np.uint8)

    histogram = np.sum(binary, axis=1)

    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(histogram)
        plt.title(f"Line Projection Histogram")
        plt.xlabel("Row Index")
        plt.ylabel("Black Pixel Count")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return histogram

def smooth_histogram(histogram):
    return gaussian_filter1d(histogram, sigma=5)


def count_peaks(histogram, height_threshold=25, distance=30, plot=True):
    peaks, _ = find_peaks(histogram, height=height_threshold, distance=distance)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(histogram, label="Histogram")
        plt.plot(peaks, histogram[peaks], "rx", label="Peaks")
        plt.title("Line Projection Histogram with Peaks")
        plt.xlabel("Index")
        plt.ylabel("Black Pixel Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return len(peaks), peaks

def cluster_boxes(box_centers, n_clusters):
    box_ids = [box_id for box_id, y in box_centers]
    y_coords = np.array([[y] for _, y in box_centers])

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    labels = kmeans.fit_predict(y_coords)

    clusters = {}
    for label, box_id in zip(labels, box_ids):
        clusters.setdefault(label, []).append(box_id)

    return clusters

def mask_bounding_boxes(image, stats, box_ids, padding=0):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for box_id in box_ids:
        x = stats[box_id, cv2.CC_STAT_LEFT] - padding
        y = stats[box_id, cv2.CC_STAT_TOP] - padding
        w = stats[box_id, cv2.CC_STAT_WIDTH] + 2 * padding
        h = stats[box_id, cv2.CC_STAT_HEIGHT] + 2 * padding

        x = max(0, x)
        y = max(0, y)
        x_end = min(image.shape[1], x + w)
        y_end = min(image.shape[0], y + h)

        cv2.rectangle(mask, (x, y), (x_end, y_end), 255, -1)

    return cv2.bitwise_and(image, image, mask=cv2.merge([mask]*3))

def crop_to_content(image, padding=5):
    image = image.copy()
    coords = cv2.findNonZero(image)

    x, y, w, h = cv2.boundingRect(coords)

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, image.shape[1])
    y2 = min(y + h + padding, image.shape[0])

    return image[y1:y2, x1:x2]

if __name__ == "__main__":
    for image_path in image_paths:
        file_name = image_path.split(".")[0]
        image_path = "image-data/" + image_path

        # Read and preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (1024, 1024))
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Line projection histogram
        hist = line_projection_histogram(image, axis='horizontal', show_plot=False)
        smoothed_hist = smooth_histogram(hist)
        peaks, _ = count_peaks(smoothed_hist, height_threshold=5, distance=10, plot=False)

        # Erode image to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.erode(image, kernel, iterations=1)

        # Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=cv2.CV_32S)

        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        y_coords = []
        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cx, cy) = centroids[i]

            y_coords.append((i, cy))

            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.circle(image, (int(cx), int(cy)), 2, (0, 0, 255), -1)  

        clusters = cluster_boxes(y_coords, peaks)
        for i in clusters.items():
            print(i)
            masked_image = mask_bounding_boxes(image, stats, i[1], padding=2)
            cropped_image = crop_to_content(masked_image, padding=2)

            os.makedirs('image_crops/'+file_name, exist_ok=True)
            cv2.imwrite(f"image_crops/{file_name}/crop_{i[0]}.png", cropped_image)
