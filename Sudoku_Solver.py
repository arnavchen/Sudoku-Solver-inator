# Arnav Chennamaneni

import cv2
import numpy as np
import pytesseract

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_grid(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assume largest contour is the grid
    grid_contour = max(contours, key=cv2.contourArea)
    return grid_contour

def recognize_digits(grid_img):
    config = '--psm 6 outputbase digits'
    digits = pytesseract.image_to_string(grid_img, config=config)
    return digits

def solve_sudoku(board):
    # Your backtracking algorithm implementation here
    pass

