# Sudoku-Solver-inator

import argparse  # Import argparse module for command-line argument parsing
import cv2  # Import OpenCV library
import numpy as np  # Import NumPy library
from tensorflow.keras.models import load_model  # Import function to load Keras model
import imutils  # Import imutils library for convenience functions
from solver_functions import *  # Import functions from custom Sudoku solver module

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Sudoku Solver')
parser.add_argument('image_path', type=str, help='Path to the input Sudoku image')
args = parser.parse_args()

classes = np.arange(0, 10)  # Array of possible digit classes (0-9)

model = load_model('model-OCR.h5')  # Load the pre-trained OCR model
print(model.summary())  # Print model summary to console
input_size = 48  # Size of input images expected by the OCR model

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

def split_boxes(board):
    """
    Splits the Sudoku board into 81 individual cells.

    Parameters:
    - board: Sudoku board image.

    Returns:
    - boxes: List of 81 individual cells, each resized and normalized.
    """
    rows = np.vsplit(board, 9)  # Split the board into 9 rows
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)  # Split each row into 9 columns (cells)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size)) / 255.0  # Resize each cell and normalize
            boxes.append(box)
    cv2.destroyAllWindows()
    return boxes

def displayNumbers(img, numbers, color=(0, 100, 0)):
    """
    Displays the solved numbers on the Sudoku board image.

    Parameters:
    - img: Sudoku board image.
    - numbers: List of solved numbers to be displayed.
    - color: Color of the displayed numbers.

    Returns:
    - img: Image with solved numbers displayed.
    """
    W = int(img.shape[1] / 9)  # Width of each cell
    H = int(img.shape[0] / 9)  # Height of each cell
    for i in range(9):
        for j in range(9):
            if numbers[(j * 9) + i] != 0:  # Display numbers that are not zero (solved numbers)
                cv2.putText(img, str(numbers[(j * 9) + i]), (i * W + int(W / 2) - int((W / 4)), int((j + 0.7) * H)), 
                            cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)  # Draw text on the image
    return img

# Read input image
img = cv2.imread(args.image_path)

# Extract Sudoku board from input image
board, location = find_board(img)

# Convert board to grayscale for further processing
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
rois = split_boxes(gray)  # Split Sudoku board into individual cells
rois = np.array(rois).reshape(-1, input_size, input_size, 1)  # Reshape for model input

# Use OCR model to predict numbers in each cell
prediction = model.predict(rois)
predicted_numbers = []

# Map predictions to actual digits using the defined classes
for i in prediction:
    index = (np.argmax(i))  # Get index of the maximum value (predicted digit)
    predicted_number = classes[index]  # Map index to actual digit
    predicted_numbers.append(predicted_number)

board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)  # Reshape predictions into Sudoku grid

# Solve the Sudoku board
try:
    solved_board_nums = get_board(board_num)

    # Create a binary array indicating solved (1) and unsolved (0) cells
    binArr = np.where(np.array(predicted_numbers) > 0, 0, 1)
    flat_solved_board_nums = solved_board_nums.flatten() * binArr  # Flatten and apply binary mask

    mask = np.zeros_like(board)  # Create a mask image
    solved_board_mask = displayNumbers(mask, flat_solved_board_nums)  # Display solved numbers on the mask
    inv = get_InvPerspective(img, solved_board_mask, location)  # Apply inverse perspective transformation
    combined = cv2.addWeighted(img, 0.7, inv, 1, 0)  # Combine original image with solved Sudoku overlay
    cv2.imshow("Final result", combined)  # Display final result image

except:
    print("Solution doesn't exist. Model misread digits.")  # Handle case where Sudoku cannot be solved

cv2.imshow("Input image", img)  # Display input image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all OpenCV windows
