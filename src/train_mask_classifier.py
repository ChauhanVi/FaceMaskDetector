import os
import argparse
import cv2
import glob
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from joblib import dump

def hog_features(img_gray):
    hog = cv2.HOGDescriptor(_winSize=(96,96),
                            _blockSize=(32,32),
                            _blockStride=(16,16),
                            _cellSize=(16,16),
                            _nbins=9)
    img_gray = cv2.resize(img_gray, (96,96))
    return hog.compute(img_gray).reshape(-1)

def load_images_in_dir(d):
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
        paths.extend(glob.glob(os.path.join(d, ext)))
    return paths

def build_dataset(base_dir):
    X, y = [], []
    for label_name, label in [("without_mask", 0), ("with_mask", 1)]:
        cls_dir = os.path.join(base_dir, label_name)
        for p in load_images_in_dir(cls_dir):
            img = cv2.imread(p)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            feat = hog_features(gray)
            X.append(feat); y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="root with with_mask/ and without_mask/")
    ap.add_argument("--out", default="models/mask_svm.joblib")
    args = ap.parse_args()

    X, y = build_dataset(args.data_dir)
    if len(X) == 0:
        raise SystemExit("No training images found. Put images under data/with_mask and data/without_mask")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False for sparse-like HOG vectors
        ("svm", LinearSVC())
    ])

    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    print(classification_report(yte, ypred, target_names=["NO MASK","MASK"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dump(clf, args.out)
    print(f"Saved trained model to: {args.out}")
