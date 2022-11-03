import copy
from .AngleCorrection import *


class Preprocess:
    def __init__(self, image):
        self.image = image

    @staticmethod
    def __HistogramicClear(frm):  # noisy clearing
        hist = []
        indexes = []
        for row in frm[0:int(frm.shape[0]/2), :]:
            k = list(row == 0)
            hist.append(k.count(True))

        hist = np.array(hist)[::-1]
        indexes.append(20 - np.argmin(hist))

        hist = []
        for row in frm[int(frm.shape[0]/2):, :]:
            k = list(row == 0)
            hist.append(k.count(True))

        hist = np.array(hist)
        indexes.append(20 + np.argmin(hist))

        if len(indexes) > 1:
            frm = frm[indexes[0]:indexes[1], :]

        returned = np.ones((frm.shape[0]+6, frm.shape[1]+6), dtype=np.uint8) * 255
        returned[1:frm.shape[0]+1, 1:frm.shape[1]+1] = frm
        return returned

    @staticmethod
    def __Preprocess(img):
        cropped = img
        indexes = []
        hist = []
        for x, row in enumerate(img[0:int(img.shape[0] / 2), :]):
            k = list(row == 0)
            hist.append(k.count(True))
        hist = np.array(hist)[::-1]
        for x, h in enumerate(hist):
            if h <= 5:
                indexes.append(int(img.shape[0] / 2) - x)
                break

        hist = []
        for x, row in enumerate(img[int(img.shape[0] / 2):, :]):
            k = list(row == 0)
            hist.append(k.count(True))
        for x, h in enumerate(hist):
            if h <= 5:
                indexes.append(int(img.shape[0] / 2) + x)
                break
        if len(indexes) > 1:
            cropped = copy.copy(img[indexes[0]:indexes[1], :])

        indexes = []
        hist = []
        counter = 0
        for x, row in enumerate(cropped.T):
            k = list(row == 0)
            hist.append(k.count(True))
        for x, h in enumerate(hist):
            if h > 0:
                counter += 1
            elif counter < 5 and h <= 0:
                indexes.append(x)
                counter = 0
        p = []
        if len(indexes):
            p.append(indexes[-1])
        indexes = []
        counter = 0
        for x, h in enumerate(hist[::-1]):
            if h > 0:
                counter += 1
            elif x < 5:
                indexes.append(cropped.T.shape[0] - x)
            else:
                counter = 0
        if len(indexes):
            p.append(indexes[-1])
        returned = cropped
        if len(p) > 1:
            returned = copy.copy(cropped.T[p[0]:p[1], :])
            returned = returned.T

        rtrnd = np.ones((returned.shape[0] + 6, returned.shape[1] + 6), dtype=np.uint8) * 255
        rtrnd[2:returned.shape[0] + 2, 2:returned.shape[1] + 2] = returned
        return rtrnd

    def Preprocess(self, imsize=(200, 40)):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # grayscale
        zone_copy = CorrectAngle(gray)[1]  # angle correction
        blur = cv2.GaussianBlur(zone_copy, (3, 3), 0)  # blurring img
        blur = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)  # binary img
        blur = cv2.equalizeHist(blur)  # histogram equalization
        zone_copy = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # other threshold for binary img
        zone_copy = cv2.resize(zone_copy, imsize, interpolation=cv2.INTER_NEAREST)  # resize img
        for_ocr = self.__HistogramicClear(zone_copy)  # image clearing
        for_ocr = self.__Preprocess(for_ocr)
        for_ocr = cv2.resize(for_ocr, imsize, interpolation=cv2.INTER_NEAREST)  # resize img
        crop = cv2.morphologyEx(for_ocr, cv2.MORPH_ERODE, kernel=np.ones((2, 2)), iterations=1)  # apply erosion
        for_ocr = cv2.morphologyEx(crop, cv2.MORPH_DILATE, kernel=np.ones((2, 2)), iterations=1)  # apply dilation
        return for_ocr

