# Sudoku-Solver-inator
This is a Python application that uses computer vision techniques and machine learning to solve Sudoku puzzles from images. It extracts the Sudoku board from an input image, recognizes digits using an OCR (Optical Character Recognition) model, solves the puzzle using a backtracking algorithm, and overlays the solution back onto the original image.

# Features

1. Sudoku Board Detection: Automatically detects and extracts the Sudoku board from the input image using contour detection and perspective transformation.

2. Digit Recognition: Utilizes a pre-trained OCR model to recognize and predict numbers in each cell of the Sudoku board.

3. Sudoku Solving: Solves the Sudoku puzzle using a backtracking algorithm implemented in Python.

4. Visual Overlay: Displays the solved Sudoku numbers overlaid onto the original image using inverse perspective transformation.

# Dependencies

1. Python 3.x
2. OpenCV (cv2)
3. NumPy (numpy)
4. TensorFlow (tensorflow.keras)
5. imutils (imutils)

# Usage
Run the Sudoku Solver by providing the path to an input Sudoku image as a command-line argument:
   ```bash
   python Sudoku-Solver-inator.py path-to-image/img.jpg
   ```

# Acknowledgments
This project was inspired by a strange urge to try to teach myself how to use computer vision properly and give me a way to have a chance against my girlfriend (I can't guarantee the success rate against an SO).
Special thanks to contributors and open source libraries used in this project.
