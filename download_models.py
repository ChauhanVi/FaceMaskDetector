import os
import urllib.request

MODELS = [
    # OpenCV DNN face detector – deploy prototxt
    ("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
     "models/deploy.prototxt"),
    ("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
     "models/res10_300x300_ssd_iter_140000.caffemodel"),
]
os.makedirs("models", exist_ok=True)

def download(url, dest):
    if os.path.exists(dest):
        print(f"[OK] Already exists: {dest}")
        return
    print(f"[DL] {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] Saved: {dest}")
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")

if __name__ == "__main__":
    for url, dest in MODELS:
        download(url, dest)
    print("\nAll models attempted (some may have failed).")
