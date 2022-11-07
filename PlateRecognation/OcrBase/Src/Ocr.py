import pickle
import cv2
from .Segmentors import *
from .Constants import *
from .Preprocess import *


class Ocr:
    def __init__(self, image):
        self.image = image
        self.info = ""
        self.preprocessor = Preprocess(image=self.image)

    def get_string(self):
        custom_ocr_string = ""
        probas = []
        ocr_model = pickle.load(open(OCR_CLASSIFIER_PATH, 'rb'))
        self.image = self.preprocessor.Preprocess(imsize=(300, 50))
        segmente_images = HistogramSegmentation(self.image)
        print(len(segmente_images))
        for key, value in segmente_images.items():
            if value.size and value.shape[1] > 8:
                value = cv2.resize(value, (30, 60), interpolation=cv2.INTER_AREA)
                # TODO:: image preprocess and classification model...
                cv2.imshow("image", value)
                cv2.waitKey()
                im = np.reshape(value, (60 * 30))
                label = ocr_model.predict([im])
                print(label)
                probas.append(ocr_model.predict_proba([im]))
                custom_ocr_string += str(label[0][1])

        proba = 0
        if len(probas):
            for k in np.array(probas):
                proba += np.max(k)
            proba = proba / len(probas)

        self.info = f"Character Segmented in Image:{len(segmente_images)}\nString:{custom_ocr_string[-5:]}\nConfidence:{proba}"
        return custom_ocr_string, self.image

    def __str__(self):
        print(self.info)
