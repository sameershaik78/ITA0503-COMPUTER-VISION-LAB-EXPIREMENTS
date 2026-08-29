import cv2
import numpy as np

# Read image
image = cv2.imread("sample.jpg")   # Change name if needed

# Check if image loaded
if image is None:
    print("Error: Image not found!")
    exit()

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range (example: green color)
lower = np.array([35, 40, 40])
upper = np.array([85, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower, upper)

# Invert mask (to get background)
mask_inv = cv2.bitwise_not(mask)

# Extract background
background = cv2.bitwise_and(image, image, mask=mask_inv)

# Show output
cv2.imshow("Original Image", image)
cv2.imshow("Foreground Mask", mask)
cv2.imshow("Background Image", background)

cv2.waitKey(0)
cv2.destroyAllWindows()
