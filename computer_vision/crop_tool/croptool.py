import cv2
import numpy as np
import os

def crop_white_box(image_path, padding=5, debug=False):
    # Step 1: Read the image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Apply binary thresholding (detect bright areas)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    if debug:
        cv2.imshow("Thresholded Image", thresh)

    # Step 4: Find contours to locate the white box
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Find the largest contour (assumed to be the white box)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add a small padding if needed
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image.shape[1] - x)
    h = min(h + 2 * padding, image.shape[0] - y)

    # Step 6: Crop the image to the white box
    cropped_image = image[y:y+h, x:x+w]

    # Display the cropped image for debugging
    if debug:
        cv2.imshow("Cropped White Box", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Step 7: Save the cropped image
    directory, filename = os.path.split(image_path)
    name, ext = os.path.splitext(filename)
    cropped_filename = f"{name}_cropped{ext}"
    cropped_path = os.path.join(directory, cropped_filename)

    cv2.imwrite(cropped_path, cropped_image)
    print(f"Cropped image saved as: {cropped_path}")

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
    cropped_filename = f"{name}_cropped2{ext}"
    cropped_path = os.path.join(directory, cropped_filename)
    
    cv2.imwrite(cropped_path, cropped_image)
    
    print(f"Cropped image saved as: {cropped_filename}")


# Example usage
image_path = "computer_vision\crop_tool\IMG_2563_cropped.JPG"
#crop_white_box(image_path)
crop_black_box(image_path)


image_path = "computer_vision\crop_tool\crop_test.png"
crop_white_box(image_path)
crop_black_box(image_path)

image_path = "computer_vision\crop_tool\IMG_2558(1)_cropped.JPG"
#crop_white_box(image_path)
crop_black_box(image_path)


# Example usage
#image_path = 'computer_vision\crop_tool\crop_test_cropped.png'
# rotate_image(image_path)