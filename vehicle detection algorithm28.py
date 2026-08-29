import cv2

cap = cv2.VideoCapture("traffic.mp4")

bg_sub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = bg_sub.apply(frame)

    cv2.imshow("Vehicle Detection", fgmask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
