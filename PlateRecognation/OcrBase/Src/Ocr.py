import pickle
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
        self.image = self.preprocessor.Preprocess(imsize=(200, 40))
        segmente_images = HistogramSegmentation(self.image)
        print(len(segmente_images))
        for key, value in segmente_images.items():
            if value.size:
                value = cv2.resize(value, (20, 60), interpolation=cv2.INTER_NEAREST)
                # cv2.imwrite(f"{datetime.datetime.now().strftime('%d%m%H%M%S%f')}.jpg", value)
                im = np.reshape(value, (60 * 20))
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
        return custom_ocr_string[-5:], self.image

    def __str__(self):
        print(self.info)
