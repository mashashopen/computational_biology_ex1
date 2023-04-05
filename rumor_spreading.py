import random
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw


class Human:
    def __init__(self, state, believe_rumor):
        self.state = state
        self.believe_rumor = believe_rumor

    def get_state(self):
        return self.state

    def set_state(self, new_state):
        self.state = new_state

    def set_believer(self):
        self.believe_rumor = True

    def will_pass(self):
        return self.believe_rumor


# Define the probabilities for each state
P_S1 = 1
P_S2 = 2/3
P_S3 = 1/3
P_S4 = 0

prob = np.array([P_S1, P_S2, P_S3, P_S4])
prob = prob / prob.sum()

# Define the grid size
GRID_SIZE = (100, 100)

# Define a function to get the neighbors of a cell
def get_neighbors(grid, x, y):
    neighbors = []
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if i == x and j == y:
                continue
            if i >= 0 and j >= 0 and i < GRID_SIZE[0] and j < GRID_SIZE[1]:
                neighbors.append(grid[i][j])
    return neighbors


def upgrade_state(current_state):
    if current_state == "S1":
        return "S1"
    if current_state == "S2":
        return "S1"
    if current_state == "S3":
        return "S2"
    if current_state == "S4":
        return "S3"

def return_to_previous_state(current_state):
    if current_state == "S1":
        return "S2"
    if current_state == "S2":
        return "S3"
    if current_state == "S3":
        return "S4"
    if current_state == "S4":
        return "S4"


# Define a function to update the grid for one iteration
def update_grid(grid, last_gen_grid, last_cat_grid, L):
    new_grid = np.copy(grid)
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            neighbors = get_neighbors(grid, i, j)
            count_believer_neighbors = 0
            if new_grid[i][j]:
                if new_grid[i][j].will_pass():
                    pass

                else:
                    for neighbor in neighbors:
                        if neighbor and neighbor.will_pass():
                            count_believer_neighbors += 1
                            if new_grid[i][j].get_state() == "S1":
                                new_grid[i][j].set_believer()
                                break

                            if neighbor.get_state() == 'S2' and np.random.random() < P_S2:
                                new_grid[i][j].set_believer()
                                break

                            if neighbor.get_state() == 'S3' and np.random.random() < P_S3:
                                new_grid[i][j].set_believer()
                                break

                    if count_believer_neighbors >= 2:
                        new_grid[i][j].set_state(upgrade_state(new_grid[i][j].get_state()))

                    if last_cat_grid[i][j] > 0:
                        new_grid[i][j].set_state(return_to_previous_state(new_grid[i][j].get_state()))
                    last_cat_grid[i][j] = 0

    return new_grid, last_gen_grid, last_cat_grid


def main():
    n = 100

    #grid = [[Human(0, None, None) if random.random() > 0.5 else None for _ in range(n)]]*n

    grid = [[0] * n for _ in range(n)]

    # Iterate through the grid and set the value of each cell based on the random value and the threshold
    for i in range(n):
        for j in range(n):
            random.seed(time.time())
            prob_to_human = random.random()
            if prob_to_human <= 0.5:
                grid[i][j] = Human(random.choice(['S1', 'S2', 'S3', 'S4']), False)

            else:
                grid[i][j] = None


    starter_x = random.randint(0, n-1)
    starter_y = random.randint(0, n-1)
    try:
        while not grid[starter_x][starter_y]:
            starter_x = random.randint(0, n)
            starter_y = random.randint(0, n)
    except:
        print(starter_x, starter_y)

    grid[starter_x][starter_y].set_believer()

    print("the starter is: ", starter_x, starter_y)




    # Define a dictionary that maps each state to a color
    state_colors = {True: 'purple', False: 'green'}

    # Create a colormap from the state colors dictionary
    cmap = ListedColormap([state_colors[state] for state in [True, False]])

    passed_rumor_generations = {}
    L = 10
    # Run the simulation for a certain number of iterations
    NUM_ITERATIONS = 100
    last_gen_grid = np.zeros(GRID_SIZE, dtype=int)
    last_cat_grid = np.zeros(GRID_SIZE, dtype=int)
    frames = []
    for i in range(NUM_ITERATIONS):
        # Update the grid
        grid, last_gen_grid, last_cat_grid = update_grid(grid, last_gen_grid, last_cat_grid, L)

        # Create an image of the grid using the state colors and add it to the frames list
        image = Image.new('RGB', GRID_SIZE)
        draw = ImageDraw.Draw(image)
        for x in range(GRID_SIZE[0]):
            for y in range(GRID_SIZE[1]):
                if grid[x][y]:
                    #print(x, y, state_colors[grid[x][y].get_state()], grid[x][y].get_state())
                    draw.rectangle((x, y, x, y), fill=state_colors[grid[x][y].will_pass()])
                    #image.show()

        frames.append(np.array(image))

        # Create an animation from the frames and display it
    fig = plt.figure(figsize=(8, 8))
    animation = FuncAnimation(fig, lambda i: plt.imshow(frames[i], cmap=cmap), frames=len(frames), interval=100)
    plt.show()


if __name__ == "__main__":
    main()
