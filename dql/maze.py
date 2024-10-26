import numpy as np
from random import shuffle

def generate_maze(height, width):
    # Initialize maze with all walls
    maze = np.ones((height, width), dtype=np.int8)
    
    # Initialize sets for cells (using only odd coordinates for cells)
    sets = {}
    
    # Calculate actual cell positions accounting for walls
    for y in range(1, height-1, 2):
        for x in range(1, width-1, 2):
            sets[(y, x)] = (y, x)
            maze[y, x] = 0  # Mark cells as passages
    
    def find(cell):
        if sets[cell] != cell:
            sets[cell] = find(sets[cell])
        return sets[cell]
    
    def union(cell1, cell2):
        root1, root2 = find(cell1), find(cell2)
        if root1 != root2:
            sets[root2] = root1
            return True
        return False
    
    # Get all possible walls between cells
    walls = []
    for y in range(1, height-1, 2):
        for x in range(1, width-1, 2):
            # Add horizontal walls if not at last column
            if x + 2 < width-1:
                walls.append((y, x, y, x+2))
            # Add vertical walls if not at last row
            if y + 2 < height-1:
                walls.append((y, x, y+2, x))
                
    # Randomize walls
    shuffle(walls)
    
    # Remove walls to create the maze
    for y1, x1, y2, x2 in walls:
        if union((y1, x1), (y2, x2)):
            # Remove the wall between cells
            wall_y = (y1 + y2) // 2
            wall_x = (x1 + x2) // 2
            maze[wall_y, wall_x] = 0
    
    return maze

# Example usage and testing
def test_maze():
    maze = generate_maze(10, 10)
    # Verify maze dimensions
    assert maze.shape == (10, 10), f"Wrong size: {maze.shape}"
    # Verify outer walls
    assert np.all(maze[0,:] == 1), "Top wall missing"
    assert np.all(maze[-1,:] == 1), "Bottom wall missing"
    assert np.all(maze[:,0] == 1), "Left wall missing"
    assert np.all(maze[:,-1] == 1), "Right wall missing"
    return maze

# Helper function to visualize the maze
def print_maze(maze):
    for row in maze:
        print(''.join(['██' if cell else '  ' for cell in row]))

maze = test_maze()
print_maze(maze)