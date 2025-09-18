import cv2
import argparse
from utils.face_detector import CaffeFaceDetector
from utils.mask_classifier import MaskClassifier
from utils.drawing import draw_box_with_label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--save", default="", help="path to save output image (optional)")
    ap.add_argument("--face_conf", type=float, default=0.5)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    face_net = CaffeFaceDetector(conf_threshold=args.face_conf)
    clf = MaskClassifier()

    boxes, confs = face_net.detect(img)
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue
        label, p = clf.predict(face)
        color = (0, 200, 0) if label == "MASK" else (0, 0, 255)
        draw_box_with_label(img, box, label, p, color)

    cv2.imshow("Face Mask Detection (Image)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.save:
        cv2.imwrite(args.save, img)
        print(f"Saved: {args.save}")

if __name__ == "__main__":
    main()
