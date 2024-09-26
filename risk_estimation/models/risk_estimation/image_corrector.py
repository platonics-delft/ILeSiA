
import cv2
from skimage import exposure
import numpy as np

from video_embedding.utils import load, set_session

# Filters for achieving invariance to surrounding lighting conditions
IMAGE_HISTOGRAM_EQUALIZATION = True
CLAHE = True
COLOR_MATCHING = True
LOGARITHMIC_TRANSFORM = True
GAMMA_CORRECTION = True

CLAHE_TILE_GRID= (8, 8)
CLAHE_CLIP_LIMIT = 2.0

GAMMA = 2.2

def correct_image(img, img_reference = None):

    if IMAGE_HISTOGRAM_EQUALIZATION:
        img = cv2.equalizeHist(img)
    if CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        img = clahe.apply(img)
    if COLOR_MATCHING:
        img = exposure.match_histograms(img, img_reference)
    if LOGARITHMIC_TRANSFORM:
        log_image = np.log1p(img.astype(np.float64))
        log_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if GAMMA_CORRECTION:
        img = np.power(img / 255.0, GAMMA)
        img = np.uint8(img * 255)

    return img

if __name__ == '__main__':
    set_session("lighting_test_session")
    data = load("peg_door_trial_7")
    img = data['img'][0].copy()
    
    data = load("peg_door_trial_1")
    img_reference = data['img'][0]
    
    cv2.imshow("Image before", img)
    cv2.waitKey(0)
    cv2.imshow("Image reference", img_reference)
    cv2.waitKey(0)

    img = correct_image(img, img_reference)

    cv2.imshow("Image after", img)
    cv2.waitKey(0)
    cv2.waitKey(0)

    