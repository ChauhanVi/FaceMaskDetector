import cv2

def letterbox(image, new_size=(300, 300)):
    h, w = image.shape[:2]
    target_w, target_h = new_size
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh))
    canvas = cv2.copyMakeBorder(resized, 0, target_h - nh, 0, target_w - nw, cv2.BORDER_CONSTANT, value=(0,0,0))
    return canvas, scale
