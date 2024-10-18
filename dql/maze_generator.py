import random
import numpy as np

# Initialize constants
WIDTH = 10
HEIGHT = 10
WALL = 1
PATH = 0

# Maze generation using Depth-First Search (DFS)
def generate_maze(width, height):
    # Create a maze filled with walls
    maze = np.ones((height, width), dtype=int)
    
    def is_valid(x, y):
        return 1 <= x < height-1 and 1 <= y < width-1  # Avoid borders
    
    def neighbors(x, y):
        # Define possible moves (up, down, left, right)
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        random.shuffle(directions)  # Shuffle for randomized maze
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and maze[nx, ny] == WALL:
                yield (nx, ny, x + dx // 2, y + dy // 2)  # return next and wall between
    
    def dfs(x, y):
        maze[x, y] = PATH
        for nx, ny, wx, wy in neighbors(x, y):
            if maze[nx, ny] == WALL:
                maze[wx, wy] = PATH  # Remove the wall
                dfs(nx, ny)
    
    # Start DFS from a random point inside the maze
    start_x, start_y = random.choice(range(1, height-1, 2)), random.choice(range(1, width-1, 2))
    dfs(start_x, start_y)
    return maze, (start_x, start_y)

# Add goal and subgoal
def add_goal_and_subgoal(maze):
    # Find empty cells for placing goal and subgoal
    empty_cells = list(zip(*np.where(maze == PATH)))
    
    # Select two distinct random positions
    goal_pos = random.choice(empty_cells)
    subgoal_pos = random.choice([pos for pos in empty_cells if pos != goal_pos])
    
    # Mark the goal and subgoal (for example, as 3 and 2 respectively)
    maze[goal_pos] = 3  # Goal
    maze[subgoal_pos] = 2  # Subgoal
    
    return goal_pos, subgoal_pos

# Generate and display the maze
maze, start = generate_maze(WIDTH, HEIGHT)
goal, subgoal = add_goal_and_subgoal(maze)

print("Generated Maze:")
print(maze)
print(f"Start Position: {start}")
print(f"Subgoal Position: {subgoal}")
print(f"Goal Position: {goal}")
