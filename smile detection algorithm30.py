import cv2

def detect_smile(image_path):

    # Load cascades
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found!")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces (more sensitive)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:

        # Draw face box (blue)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Take only LOWER HALF of face (mouth area)
        roi_gray = gray[y + int(h/2): y + h, x: x + w]
        roi_color = image[y + int(h/2): y + h, x: x + w]

        # Detect smiles (high sensitivity)
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(15, 15)
        )

        # Draw smile boxes (green)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(
                roi_color,
                (sx, sy),
                (sx + sw, sy + sh),
                (0, 255, 0),
                2
            )

    # Show output
    cv2.imshow("Smile Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run
detect_smile("laugh.jpg")
