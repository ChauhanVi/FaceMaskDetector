import cv2
import numpy as np
import os

class CaffeFaceDetector:
    def __init__(self, prototxt="models/deploy.prototxt",
                 weights="models/res10_300x300_ssd_iter_140000.caffemodel",
                 conf_threshold=0.5):
        if not (os.path.exists(prototxt) and os.path.exists(weights)):
            raise FileNotFoundError("Face detector model files not found. Run `python download_models.py` first.")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        confs = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >= self.conf_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
                    confs.append(float(confidence))
        return boxes, confs
