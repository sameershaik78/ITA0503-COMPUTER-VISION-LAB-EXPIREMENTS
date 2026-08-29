import cv2
import pytesseract

# Set tesseract path (change if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_video(video_path):

    cap = cv2.VideoCapture(video_path)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Improve text clarity
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

        # Extract text
        text = pytesseract.image_to_string(gray)

        if text.strip() != "":
            print("Detected Text:")
            print(text)
            print("---------------")

        # Show video
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Run
extract_text_from_video("input.mp4")
