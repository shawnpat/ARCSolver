import numpy as np


# Define the transformation function according to the rules
def transform_grid(grid):
    grid = np.array(grid)
    output = np.zeros(grid.shape, dtype=int)
    rows, cols = grid.shape

    for r in range(rows):
        for c in range(cols):
            current_value = grid[r, c]
            neighbors = []

            # Gather values of neighbors if they exist (up, down, left, right)
            if r > 0:
                neighbors.append(grid[r - 1, c])
            if r < rows - 1:
                neighbors.append(grid[r + 1, c])
            if c > 0:
                neighbors.append(grid[r, c - 1])
            if c < cols - 1:
                neighbors.append(grid[r, c + 1])

            # Apply transformation rules
            if current_value > 0:
                output[r, c] = sum([n for n in neighbors if n > 0]) // 4
            elif current_value == 0 and any(n > 0 for n in neighbors):
                output[r, c] = max(neighbors) // 2

    return output.tolist()


# Define the test input grid
test_input_grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0],
    [0, 0, 8, 0, 0, 8, 0, 0, 8, 8, 8, 0, 0, 8, 0, 8, 0],
    [0, 0, 8, 8, 8, 8, 8, 8, 8, 0, 8, 0, 0, 8, 8, 8, 0],
    [0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0, 0, 8, 0, 8, 0],
    [0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0],
    [0, 0, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 8, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8],
]

# Apply the transformation to the test input grid
test_output_grid = transform_grid(test_input_grid)

# Display the predicted output grid
for row in test_output_grid:
    print(row)
