import cv2
import numpy as np

# Read image
image = cv2.imread("sample.jpg")   # Change if needed

# Check image
if image is None:
    print("Error: Image not found!")
    exit()

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range (example: green object)
lower = np.array([35, 40, 40])
upper = np.array([85, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower, upper)

# Extract foreground
foreground = cv2.bitwise_and(image, image, mask=mask)

# Show results
cv2.imshow("Original Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Foreground Image", foreground)

cv2.waitKey(0)
cv2.destroyAllWindows()
