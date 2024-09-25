import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
CELL_SIZE = 40

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Create the screen object
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Maze definition (1 = wall, 0 = empty space)
# maze = [
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
#     [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#     [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ]

maze = np.random.rand(15, 15)
int_transform = lambda x: 1 if x > 0.5 else 0
int_transform = np.vectorize(int_transform)
maze = int_transform(maze)
maze[0, :] = 1
maze[-1, :] = 1
maze[:, 0] = 1
maze[:, -1] = 1

# Player starting position
player_x = 1
player_y = 1

goal_x = 13
goal_y = 5

def draw_maze():
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            color = BLACK if maze[row][col] == 1 else WHITE
            pygame.draw.rect(screen, color, pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def draw_player(x, y):
    pygame.draw.rect(screen, GREEN, pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def draw_goal(x, y):
    pygame.draw.rect(screen, BLUE, pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Player movement handling
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        if maze[player_y][player_x - 1] == 0:
            player_x -= 1
    if keys[pygame.K_RIGHT]:
        if maze[player_y][player_x + 1] == 0:
            player_x += 1
    if keys[pygame.K_UP]:
        if maze[player_y - 1][player_x] == 0:
            player_y -= 1
    if keys[pygame.K_DOWN]:
        if maze[player_y + 1][player_x] == 0:
            player_y += 1

    # Clear the screen
    screen.fill(WHITE)

    # Draw the maze, player, and goal
    draw_maze()
    draw_player(player_x, player_y)
    draw_goal(goal_x, goal_y)

    # Check win condition
    if player_x == goal_x and player_y == goal_y:
        print("You win!")
        pygame.quit()
        sys.exit()

    # Update the display
    pygame.display.flip()

    # Frame rate control
    pygame.time.Clock().tick(10)
