# extraction_functions

import cv2  # Import OpenCV library
import numpy as np  # Import NumPy library
from tensorflow.keras.models import load_model  # Import function to load Keras model
import imutils  # Import imutils library for convenience functions


def get_perspective(img, location, height=900, width=900):
    """
    Applies perspective transformation to extract a specific region from the image.

    Parameters:
    - img: Input image from which region is to be extracted.
    - location: Coordinates of the region to be extracted (vertices of a quadrilateral).
    - height: Height of the extracted region.
    - width: Width of the extracted region.

    Returns:
    - result: Extracted region after perspective transformation.
    """
    pts1 = np.float32([location[0], location[3], location[1], location[2]])  # Coordinates of the region in the input image
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # Coordinates of the region in the output image

    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))  # Apply the transformation
    return result

def get_InvPerspective(img, masked_num, location, height=900, width=900):
    """
    Applies inverse perspective transformation to overlay solved numbers back onto the original image.

    Parameters:
    - img: Original input image.
    - masked_num: Masked image containing solved numbers.
    - location: Coordinates of the region where numbers are to be overlaid.
    - height: Height of the original region.
    - width: Width of the original region.

    Returns:
    - result: Image with solved numbers overlaid back onto the original perspective.
    """
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # Coordinates of the original region
    pts2 = np.float32([location[0], location[3], location[1], location[2]])  # Coordinates of the transformed region

    # Apply inverse perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))  # Apply the transformation
    return result

def find_board(img):
    """
    Finds the Sudoku board within the input image using contour detection.

    Parameters:
    - img: Input image containing the Sudoku puzzle.

    Returns:
    - result: Extracted Sudoku board after perspective transformation.
    - location: Coordinates of the Sudoku board within the input image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)  # Apply bilateral filtering to reduce noise
    edged = cv2.Canny(bfilter, 30, 180)  # Apply Canny edge detection
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = imutils.grab_contours(keypoints)  # Extract contours using imutils

    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)  # Draw contours on a copy of the original image

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]  # Sort contours by area to find largest rectangular contour
    location = None

    # Find rectangular contour (Sudoku board)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)  # Approximate the contour to a simpler polygon
        if len(approx) == 4:  # If the contour has 4 vertices, it's likely the Sudoku board
            location = approx
            break

    result = get_perspective(img, location)  # Extract the Sudoku board using perspective transformation
    return result, location