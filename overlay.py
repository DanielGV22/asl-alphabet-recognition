import cv2

def draw_bbox(frame, x0, y0, x1, y1, text):
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(frame, text, (x0, max(0, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def draw_hud(frame, lines, x=10, y=25, dy=22):
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (x, y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
