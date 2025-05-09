import cv2
import os
from src.segment.line import mask_bounding_boxes, crop_to_content

def run(args):
    image_paths = os.listdir("outputs/line-crops")

    for image_path in image_paths:
        image_names = os.listdir(f"outputs/line-crops/" + image_path)

        for image_name in image_names:
            path = f"outputs/line-crops/{image_path}/{image_name}"
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8, ltype=cv2.CV_32S)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                
                if area < 50:
                    continue

                mask = mask_bounding_boxes(image, stats, [i], padding=2)
                img = crop_to_content(mask, padding=2)

                os.makedirs(f"outputs/char-crops/{image_path}/{image_name}", exist_ok=True)
                cv2.imwrite(f"outputs/char-crops/{image_path}/{image_name}/char_{i}.png", img)       