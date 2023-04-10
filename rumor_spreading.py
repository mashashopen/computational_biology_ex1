import random
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw


# Define the probabilities for each state
P_S1 = 1
P_S2 = 2/3
P_S3 = 1/3
P_S4 = 0



N = 100
L = 20
P = 0.5
# Define the grid size
GRID_SIZE = (N, N)


class Human:
    def __init__(self, state, believe_rumor):
        self.state = state
        self.believe_rumor = believe_rumor
        self.L = L

    def get_state(self):
        return self.state

    def set_state(self, new_state):
        self.state = new_state

    def set_believer(self):
        self.believe_rumor = True
        self.L = L

    def is_believer(self):
        return self.believe_rumor

    def is_passing(self):
        return self.is_believer() and (self.L == 0 or self.L == L)

    def update_gen(self):
        self.L -= 1

    def get_L(self):
        return self.L




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
def update_grid(grid):
    new_grid = np.copy(grid)
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            neighbors = get_neighbors(grid, i, j)
            count_believer_neighbors = 0
            if grid[i][j]:
                if grid[i][j].is_believer() and not grid[i][j].is_passing():    # already passed the rumor
                    # in previous generation. should be blocked and continue counting generations
                    new_grid[i][j].update_gen()
                    break

                else:
                    for neighbor in neighbors:
                        if neighbor and neighbor.is_passing():
                            count_believer_neighbors += 1

                            if grid[i][j].get_state() == "S1":
                                new_grid[i][j].set_believer()
                                break

                            if grid[i][j].get_state() == 'S2' and np.random.random() < P_S2:
                                new_grid[i][j].set_believer()
                                break

                            if grid[i][j].get_state() == 'S3' and np.random.random() < P_S3:
                                new_grid[i][j].set_believer()
                                break

                    if count_believer_neighbors >= 2:
                        new_grid[i][j].set_state(upgrade_state(new_grid[i][j].get_state()))

                if new_grid[i][j].get_L() <= 0:
                    new_grid[i][j].set_state(return_to_previous_state(new_grid[i][j].get_state()))

    return new_grid


def main():

    grid = [[0] * GRID_SIZE[0] for _ in range(GRID_SIZE[1])]    # init array to put humans in

    # Iterate through the grid and set the value of each cell based on the random value and the threshold
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            random.seed(time.time())
            prob_to_human = random.random()
            if prob_to_human <= P:
                grid[i][j] = Human(random.choice(['S1', 'S2', 'S3', 'S4']), False)  # init Human object in each cell
                # consider each human as non-believer yet

            else:
                grid[i][j] = None   # not a human

    # starter position

    starter_x = random.randint(0, N-1)
    starter_y = random.randint(0, N-1)
    try:
        while not grid[starter_x][starter_y]:
            starter_x = random.randint(0, N-1)
            starter_y = random.randint(0, N-1)
    except:
        print(starter_x, starter_y)

    first_spreader = grid[starter_x][starter_y]
    first_spreader.set_believer()

    print("the starter is: ", starter_x, starter_y)

    # Define a dictionary that maps each state to a color
    state_colors = {
        (True, 'S1'): 'purple',
        (True, 'S2'): 'purple',
        (True, 'S3'): 'purple',
        (True, 'S4'): 'purple',
        (False, 'S1'): 'green',
        (False, 'S2'): 'white',
        (False, 'S3'): 'pink',
        (False, 'S4'): 'red'
    }
    # Create a colormap from the state colors dictionary
    cmap = ListedColormap(list(set(state_colors.values())))

    # Run the simulation for a certain number of iterations
    NUM_ITERATIONS = 100

    frames = []
    for i in range(NUM_ITERATIONS):

        # Create an image of the grid using the state colors and add it to the frames list
        image = Image.new('RGB', GRID_SIZE)
        draw = ImageDraw.Draw(image)
        for x in range(GRID_SIZE[0]):
            for y in range(GRID_SIZE[1]):
                if grid[x][y]:
                    draw.rectangle((x, y, x, y), fill=state_colors[(grid[x][y].is_believer(), grid[x][y].get_state())])
        frames.append(np.array(image))
        # Update the grid
        grid = update_grid(grid)

        # Create an animation from the frames and display it
    fig = plt.figure(figsize=(8, 8))
    animation = FuncAnimation(fig, lambda i: plt.imshow(frames[i], cmap=cmap), frames=len(frames), interval=1000)
    plt.show()


if __name__ == "__main__":
    main()
