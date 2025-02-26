import cv2
import numpy as np
from PIL import Image
import pytesseract

# Load image
image = cv2.imread('/Users/shashwatsingh/Downloads/Nutrition.jpg')
#image = cv2.imread('/Users/shashwatsingh/Downloads/Nutrition2.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Apply Gaussian Blur to remove noise
gray = cv2.GaussianBlur(gray, (5,5), 0)

# Apply CLAHE for contrast improvement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# Apply thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save the processed image (optional)
#cv2.imwrite("processed.jpg", thresh)

# Perform OCR with custom config
custom_config = r'--oem 3 --psm 6 -l eng'
text = pytesseract.image_to_string(Image.fromarray(thresh), config=custom_config)


print("Extracted Text:")
print(text)

