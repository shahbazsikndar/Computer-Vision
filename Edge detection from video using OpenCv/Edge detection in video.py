import cv2 as cv
import numpy as np


camera = cv.VideoCapture(0)  # write filename.mp4 instead of 0 to detecrt edges from video file
while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        cv.imshow('camera', frame)

        # Apply Laplacian to reduce noise and improve edge detection
        laplacian = cv.Laplacian(frame, cv.CV_64F)
        laplacian_8u = np.uint8(laplacian)
        cv.imshow('laplacian', laplacian)
        # Apply Canny edge detector
        edges=cv.Canny(frame, 100,100)
        cv.imshow('Canny', edges)

         # Break the loop if the user presses 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
           break

    # Release the camera and close all OpenCV windows
camera.release()
cv.destroyAllWindows()