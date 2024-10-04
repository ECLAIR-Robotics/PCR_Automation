import cv2
import numpy as np
import os

def crop_black_box(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale (since black box detection is color independent)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get only black colors
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the black areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest black contour is the box we want
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Save the cropped image in the same folder with '_cropped' added to the filename
    directory, filename = os.path.split(image_path)
    name, ext = os.path.splitext(filename)
    cropped_filename = f"{name}_cropped{ext}"
    cropped_path = os.path.join(directory, cropped_filename)
    
    cv2.imwrite(cropped_path, cropped_image)
    
    print(f"Cropped image saved as: {cropped_filename}")

# Example usage
image_path = "computer_vision\crop_tool\crop_test.png"
crop_black_box(image_path)