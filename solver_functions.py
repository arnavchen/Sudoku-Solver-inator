# solver_functions

def find_empty(board):
    """
    Finds the first empty cell (represented by 0) in the Sudoku board.

    Parameters:
    - board: A 2D list (9x9 matrix) representing the Sudoku board.

    Returns:
    - Tuple (row, col): Position of the first empty cell found.
      Returns None if no empty cell is found.
    """
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col
    return None


def valid(board, num, pos):
    """
    Checks if placing 'num' in the position 'pos' of the Sudoku board is valid.

    Parameters:
    - board: A 2D list (9x9 matrix) representing the Sudoku board.
    - num: The number to be checked for validity.
    - pos: Tuple (row, col) representing the position to check.

    Returns:
    - bool: True if 'num' can be placed in 'pos' without violating Sudoku rules, False otherwise.
    """
    # Check row
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    # Check 3x3 box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True


def solve(board):
    """
    Solves the Sudoku puzzle using backtracking.

    Parameters:
    - board: A 2D list (9x9 matrix) representing the unsolved Sudoku board.
            Empty cells are represented by 0.

    Returns:
    - bool: True if the Sudoku puzzle is solved successfully, False otherwise.
    """
    find = find_empty(board)
    if not find:
        return True
    else:
        row, col = find

    for num in range(1, 10):  # Try numbers 1 through 9
        if valid(board, num, (row, col)):
            board[row][col] = num

            if solve(board):
                return True

            board[row][col] = 0  # Backtrack

    return False


def get_board(board):
    """
    Solves the Sudoku board and returns the solved board.

    Parameters:
    - board: A 2D list (9x9 matrix) representing the unsolved Sudoku board.
            Empty cells are represented by 0.

    Returns:
    - board: A 2D list (9x9 matrix) representing the solved Sudoku board.
    Raises:
    - ValueError: If no solution exists for the Sudoku puzzle.
    """
    if solve(board):
        return board
    else:
        raise ValueError("No solution exists for the Sudoku puzzle.")


