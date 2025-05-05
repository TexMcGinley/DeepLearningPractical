import os
import cv2                       # imported but not used in this snippet
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi # not used here, kept in case you add smoothing
from PIL import Image, ImageDraw

# --------------------------------------------------------------------
# iterate over all files inside the folder
image_paths = os.listdir("image-data")

def horizontal_line_segmentation(image_path, N=80):
    """
    Detects horizontal text lines via projection‑profile minima.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    N : int, optional (default = 80)
        Half‑window size for the local‑minimum test.  A row `i` is accepted
        as a valley if it is strictly lower than *all* rows between
        `i‑N` and `i+N`.

    Returns
    -------
    img_with_lines : PIL.Image (RGB)
        Copy of the input image with red staff lines drawn.
    minima : list[int]
        List of row indices that were classified as valleys.
    """
    # --- 1. load → grayscale -------------------------------------------------
    img = Image.open(image_path).convert("L")  # 'L' = 8‑bit grayscale
    img_array = np.array(img)

    # --- 2. horizontal projection profile -----------------------------------
    # Each element is the total pixel intensity of one row.
    # (Black text ≈ small value, white background ≈ large value.)
    pixel_sum_rows = img_array.sum(axis=1)

    # --- 3. scale profile to [0, 1] so thresholds are resolution‑independent --
    pixel_sum_rows_norm = (
        pixel_sum_rows - pixel_sum_rows.min()
    ) / (pixel_sum_rows.ptp() + 1e-8)

    # --- 4. find local minima inside a ±N‑row neighbourhood -----------------
    minima = []
    for i in range(N, len(pixel_sum_rows_norm) - N):
        # is row i lower than every one of its N rows above *and* below?
        lower_than_left  = all(pixel_sum_rows_norm[i] < pixel_sum_rows_norm[i - j] for j in range(1, N + 1))
        lower_than_right = all(pixel_sum_rows_norm[i] < pixel_sum_rows_norm[i + j] for j in range(1, N + 1))
        if lower_than_left and lower_than_right:
            minima.append(i)

    # --- 5. prepare an RGB image so we can draw coloured lines ---------------
    img_with_lines = img.convert("RGB")
    draw = ImageDraw.Draw(img_with_lines)

    # --- 6. draw red staff lines at the mid‑points between consecutive minima
    # (If you want the lines through the *gaps* themselves, loop over minima.)
    if len(minima) >= 2:
        for k in range(len(minima) - 1):
            midpoint = (minima[k] + minima[k + 1]) // 2
            # the colour tuple is (R, G, B)  – (255, 0, 0) = red
            draw.line([(0, midpoint), (img.width, midpoint)], fill=(255, 0, 0), width=4)

    # --- 7. show result ------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.imshow(img_with_lines)
    plt.title(f"Staff lines via projection minima  (N = {N})")
    plt.axis("off")
    plt.show()

    return img_with_lines, minima

# --------------------------------------------------------------------
# Run the detector on every image in the folder
for image_path in image_paths:
    image_path = "image-data/" + image_path
    img_with_lines, minima = horizontal_line_segmentation(image_path, N=80)
