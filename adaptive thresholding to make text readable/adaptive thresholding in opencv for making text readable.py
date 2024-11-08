import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('image1.jpeg', cv2.IMREAD_GRAYSCALE)

# 1. Simple (Normal) Thresholding
_, normal_thresh = cv2.threshold(image, 111, 255, cv2.THRESH_BINARY) # change value to get the best results

# 2. Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    image, 111, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Adaptive method
    cv2.THRESH_BINARY,              # Threshold type
    19,                             # Block size (size of neighborhood) change value to get the best results
    10                              # Constant subtracted from the mean change value to get the best results
)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Normal Thresholding', normal_thresh)
cv2.imshow('Adaptive Thresholding', adaptive_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
