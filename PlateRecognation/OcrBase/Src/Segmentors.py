import cv2
import numpy as np


def SimpleSegmentation(img):
    new_img = []
    segmente_images = {}
    count = 0
    counter = 0
    for row in img.T[::-1]:
        k = list(row == 0)
        if k.count(True) > 0:
            new_img.append(row)
        else:
            counter += 1
            try:
                if len(new_img) > 1 and counter > 1:
                    counter = 0
                    newimg_sq = np.squeeze(new_img)
                    rot_img = np.rot90(newimg_sq, 3)
                    segmente_images[count] = rot_img
                    new_img = []
                    count += 1
            except ValueError:
                pass
    returned_dict = {}
    flag = False
    for k, v in segmente_images.items():
        if flag:
            flag = False
            continue
        if v.shape[1] > 15:
            returned_dict[k] = v
        else:
            if k < len(segmente_images) - 1:
                if segmente_images[k + 1].shape[1] <= 15:
                    c = np.hstack((segmente_images[k + 1], v))
                    returned_dict[k] = c
                    flag = True
                else:
                    returned_dict[k] = v
            else:
                returned_dict[k] = v
    return dict(reversed(list(returned_dict.items())))


def HistogramSegmentation(frm):
    hist = []
    copychar = []
    indexes = []
    for row in frm.T:
        k = list(row == 0)
        hist.append(k.count(True))

    for x, p in enumerate(hist):
        if not p == 0:
            indexes.append(x)
        else:
            if len(indexes) > 0:
                copychar.append(indexes)
                indexes = []

    indexes = []
    for k, i in enumerate(copychar):
        final = i[-1]
        if len(i) > 10:
            indexes.append(i)
            if k < len(copychar) - 1:
                if copychar[k + 1][0] - final < 5 and (len(i) < 18) and len(copychar[k + 1]) < 18:
                    i[-1] = copychar[k + 1][-1]
                    copychar.pop(k + 1)
                    indexes[-1] = i

    # if len(indexes) == 7:
    #     indexes[0][-1] = indexes[1][-1]
    #     indexes.pop(1)
    returned_dict = {}
    c = 0
    for x, k in enumerate(indexes):
        im = frm.T[k[0]:k[-1], :].T
        returned_dict[c] = im
        c += 1

    return returned_dict


def CountourSegmentation(LpImg):
    if len(LpImg):  # check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        # plate_image = cv2.convertScaleAbs(LpImg[0], alpha=255.0)

        # convert to grayscale and blur the image
        # gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(LpImg, (7, 7), 0)

        # Applied inversed thresh_binary
        binary = cv2.threshold(blur, 180, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Applied dilation
        cv2.imshow("w", binary)
        cv2.waitKey()
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

        # Create sort_contours() function to grab the contour of each digit from left to right
        def sort_contours(cnts, reverse=False):
            i = 0
            boundingBoxes = [cv2.boundingRect(cn) for cn in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                key=lambda b: b[1][i], reverse=reverse))
            return cnts

        cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # creat a copy version "test_roi" of plat_image to draw bounding box
        test_roi = LpImg.copy()

        # Initialize a list which will be used to append charater image
        crop_characters = []

        # define standard width and height of character
        digit_w, digit_h = 30, 60

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
                if h / LpImg.shape[0] >= 0.5:  # Select contour which has the height larger than 50% of the plate
                    # Draw bounding box arroung digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Sperate number and gibe prediction
                    curr_num = thre_mor[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(cv2.threshold(curr_num, 127, 255, cv2.THRESH_BINARY_INV)[1])

        print("Detect {} letters...".format(len(crop_characters)))
        return crop_characters
