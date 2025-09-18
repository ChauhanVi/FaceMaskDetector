import cv2
import numpy as np
import os
from joblib import load

# If models/mask_svm.joblib exists, use it. Else use heuristic.

def _skin_mask_bgr(img):
    # Simple skin detection in YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (_, cr, cb) = cv2.split(ycrcb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))  # classic YCrCb skin range
    return skin

def _heuristic_mask_score(face_bgr):
    # Focus on lower half (where masks sit)
    h, w = face_bgr.shape[:2]
    lower = face_bgr[h//2:h, :]
    skin = _skin_mask_bgr(lower)
    skin_ratio = (skin > 0).sum() / (lower.shape[0] * lower.shape[1] + 1e-6)
    # If there's a mask, we expect *less skin* visible on lower half
    # Return probability-like score for "MASK"
    # Map skin_ratio in [0, ~0.7] -> higher means NO MASK
    score = 1.0 - min(0.7, skin_ratio) / 0.7
    return float(score)  # ~1 = mask, ~0 = no mask

class MaskClassifier:
    def __init__(self, model_path="models/mask_svm.joblib"):
        self.use_svm = os.path.exists(model_path)
        self.svm = None
        if self.use_svm:
            try:
                self.svm = load(model_path)
            except Exception:
                self.use_svm = False

    def _hog_features(self, face_gray):
        face_gray = cv2.resize(face_gray, (96, 96))
        hog = cv2.HOGDescriptor(_winSize=(96,96),
                                _blockSize=(32,32),
                                _blockStride=(16,16),
                                _cellSize=(16,16),
                                _nbins=9)
        return hog.compute(face_gray).reshape(-1)

    def predict(self, face_bgr):
        """
        Returns: label(str), prob(float) where label in {"MASK","NO MASK"}
        """
        if self.use_svm and self.svm is not None:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            feat = self._hog_features(gray)
            prob = None
            if hasattr(self.svm, "predict_proba"):
                prob = self.svm.predict_proba([feat])[0]
                p_mask = float(prob[1])  # class 1 = mask
            else:
                p_mask = float(self.svm.decision_function([feat])[0])
                p_mask = 1.0 / (1.0 + np.exp(-p_mask))
            label = "MASK" if p_mask >= 0.5 else "NO MASK"
            return label, p_mask
        else:
            p_mask = _heuristic_mask_score(face_bgr)
            label = "MASK" if p_mask >= 0.5 else "NO MASK"
            return label, p_mask
