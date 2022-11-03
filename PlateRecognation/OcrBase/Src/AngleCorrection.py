import cv2
import numpy as np
from scipy.ndimage import rotate


def CorrectAngle(image, delta=3, limit=50):
    def determine_score(arr, skew):
        data = rotate(arr, skew, reshape=False, order=0)
        histograms = np.sum(data, axis=1, dtype=float)
        scr = np.sum((histograms[1:] - histograms[:-1]) ** 2, dtype=float)
        return histograms, scr

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
    return best_angle, corrected_img
