import numpy as np
import cv2

# Create a white image with dimensions (640, 480) using NumPy
white_image = np.full((480, 640, 3), 200, dtype=np.uint8)

# Display the image using OpenCV
cv2.imshow('White Image', white_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
