import cv2

cap = cv2.VideoCapture("traffic.mp4")

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

for frame in reversed(frames):
    cv2.imshow("Reverse Slow Motion", frame)
    if cv2.waitKey(120) & 0xFF == 27:
        break

cv2.destroyAllWindows()
