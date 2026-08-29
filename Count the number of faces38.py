import cv2

def detect_and_count_faces(image_path):

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Read image
    image = cv2.imread(image_path)

    # Check if image loaded
    if image is None:
        print("Error: Image not found!")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image,
                      (x, y),
                      (x + w, y + h),
                      (0, 255, 0),
                      2)

    # Count faces
    count = len(faces)
    print("Number of faces detected:", count)

    # Show result
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call function
detect_and_count_faces("face.jpg")
