import copy
import sys
from argparse import ArgumentParser
import cv2
import torch
from psutil import virtual_memory
import util
from OcrBase.Src.Ocr import Ocr


def run(predictor):
    cap = cv2.VideoCapture(arguments.source)
    while True:
        vm = virtual_memory()
        if vm.percent > 95:
            print(f"Ram Usage is Over the %95::{vm.percent}")
            sys.exit()
        ret, frame = cap.read()
        frame_copy = copy.copy(frame)
        shape = frame.shape
        if ret:
            results = predictor(frame)
            records = results.xyxyn[0].numpy()
            records = util.choose_records(records)
            Strs = []
            if len(records) > 0:
                records = util.create_ids(records)
                for k in records:
                    zone = frame[int(k[1] * shape[0]):int(k[3] * shape[0]), int(k[0] * shape[1]):int(k[2] * shape[1])]
                    ocr = Ocr(image=zone)
                    string, im = ocr.get_string()
                    Strs.append(string)

                for index, k in enumerate(records):
                    cv2.rectangle(frame, (int(k[0] * shape[1]), int(k[1] * shape[0])), (int(k[2] * shape[1]), int(k[3] * shape[0])),
                                  (0, 255, 0), 1, cv2.FONT_HERSHEY_SIMPLEX)
                    cv2.putText(frame, str(k[-1]), (int(k[0] * shape[1]), int(k[3] * shape[0] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame, Strs[index], (int(k[0] * shape[1]), int(k[1] * shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if arguments.watch == "True":
                cv2.imshow("frame", frame)
                cv2.waitKey(1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights", help=f"Pretrained model weights",
                        default=r"best.pt")
    parser.add_argument("--yoloPath", help=f"yolov5 repo dir", default=r"ultralytics/yolov5")
    parser.add_argument("--source", help=f"Source: Rtsp url, video path, or cam (if you use camera use --source 0)",
                        default=r"video.mp4")
    parser.add_argument("--watch", help="if you want to watch frames", default="True")
    arguments = parser.parse_args()
    model = torch.hub.load(arguments.yoloPath, 'custom', path=arguments.weights, force_reload=True)
    run(predictor=model)
