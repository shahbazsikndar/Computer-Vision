import cv2 as cv

# Initialize the video capture object
cap = cv.VideoCapture('2.mp4')  # Replace 'input_video.mp4' with 0 to use webcam

# Create the background subtractor
background_subtractor = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = background_subtractor.apply(frame)

    # Optional: filter the mask to reduce noise
    # Apply morphological operations to clean up the mask
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

    # Display the original frame and the foreground mask
    cv.imshow('Original Video', frame)
    cv.imshow('Motion Detection (Foreground Mask)', fg_mask)

    # Exit loop if 'q' is pressed
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
