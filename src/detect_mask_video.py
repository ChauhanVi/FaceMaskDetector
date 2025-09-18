import cv2
import argparse
from utils.face_detector import CaffeFaceDetector
from utils.mask_classifier import MaskClassifier
from utils.drawing import draw_box_with_label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="0 for webcam or path to video")
    ap.add_argument("--face_conf", type=float, default=0.5, help="face detector confidence threshold")
    args = ap.parse_args()

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    face_net = CaffeFaceDetector(conf_threshold=args.face_conf)
    clf = MaskClassifier()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes, confs = face_net.detect(frame)

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            label, p = clf.predict(face)
            color = (0, 200, 0) if label == "MASK" else (0, 0, 255)
            draw_box_with_label(frame, box, label, p, color)

        cv2.imshow("Face Mask Detection (OpenCV DNN + Caffe)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
