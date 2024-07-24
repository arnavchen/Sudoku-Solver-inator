# Sudoku-Solver-inator
This is a Python application that uses computer vision techniques and machine learning to solve Sudoku puzzles from images. It extracts the Sudoku board from an input image, recognizes digits using an OCR (Optical Character Recognition) model, solves the puzzle using a backtracking algorithm, and overlays the solution back onto the original image.


Features
Sudoku Board Detection: Automatically detects and extracts the Sudoku board from the input image using contour detection and perspective transformation.

Digit Recognition: Utilizes a pre-trained OCR model to recognize and predict numbers in each cell of the Sudoku board.

Sudoku Solving: Solves the Sudoku puzzle using a backtracking algorithm implemented in Python.

Visual Overlay: Displays the solved Sudoku numbers overlaid onto the original image using inverse perspective transformation.

Dependencies
Python 3.x
OpenCV (cv2)
NumPy (numpy)
TensorFlow (tensorflow.keras)
imutils (imutils)

Usage
Run the Sudoku Solver by providing the path to an input Sudoku image as a command-line argument:

Acknowledgments
This project was inspired by a strange urge to try to teach myself how to use computer vision properly and give me a way to have a chance against my girlfriend (I can't guarantee the success rate against an SO).
Special thanks to contributors and open source libraries used in this project.
