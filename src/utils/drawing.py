import cv2

def draw_box_with_label(img, box, label, prob, color):
    (x1, y1, x2, y2) = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    txt = f"{label}: {prob*100:.1f}%"
    (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(0, y1 - 8)
    cv2.rectangle(img, (x1, y - th - baseline - 6), (x1 + tw + 6, y), color, -1)
    cv2.putText(img, txt, (x1 + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
