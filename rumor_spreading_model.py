import random
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw
from numpy import double

# Define the probabilities for each state
P_S1 = 1
P_S2 = 2/3
P_S3 = 1/3
P_S4 = 0

NUM_GENERATIONS = 100
N = 100
L = 10
P = 0.5
starters_num = 1
# Define the grid size
GRID_SIZE = (N, N)
human_recived = []


class Human:
    def __init__(self, state, receive_rumor=False, spread_rumor=False, num_of_gen=0, state_is_updated=False):
        self.state = state
        self.receive_rumor = receive_rumor
        self.spread_rumor = spread_rumor
        self.num_of_gen = num_of_gen
        self.state_is_updated = state_is_updated

    def set_spreader(self):
        self.spread_rumor = True

    def is_spreader(self):
        return self.spread_rumor

    def set_receiver(self):
        self.receive_rumor = True

    def is_receiver(self):
        return self.receive_rumor

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_all_fields(self):
        return self.state, self.receive_rumor, self.spread_rumor, self.num_of_gen, self.state_is_updated

    def update_num_of_gen(self, new_L):
        self.num_of_gen = new_L

    def get_next_gen(self):
        return self.num_of_gen

    def set_next_gen(self, gen):
        self.num_of_gen = gen

    def state_updated(self):
        self.state_is_updated = True

    def is_state_updated(self):
        return self.state_is_updated


# Define a function to get the Human neighbors of a Human
def get_neighbors(grid, i, j):
    neighbors = []
    for row in range(i - 1, i + 2):
        for col in range(j - 1, j + 2):
            if row == i and col == j:
                continue
            if 0 <= row < GRID_SIZE[0] and 0 <= col < GRID_SIZE[1]:
                if grid[row][col]:
                    neighbors.append((row, col))
    return neighbors


def build_test_grid():
    grid = [[0] * GRID_SIZE[0] for _ in range(GRID_SIZE[1])]

    return grid


# if current human is a spreader decide if his neighbors will believe him or not.
def decide_if_neighbor_will_spread(prev_gen_grid, next_gen_grid, row, col, num_of_gen):

    prev_neighbor = prev_gen_grid[row][col]
    next_neighbor = next_gen_grid[row][col]

    if prev_neighbor.get_state() == 'S1' and random.random() < P_S1:
        next_neighbor.set_spreader()
        next_neighbor.set_next_gen(num_of_gen + 1)

    elif prev_neighbor.get_state() == 'S2' and random.random() < P_S2:
        next_neighbor.set_spreader()
        next_neighbor.set_next_gen(num_of_gen + 1)

    elif prev_neighbor.get_state() == 'S3' and random.random() < P_S3:
        next_neighbor.set_spreader()
        next_neighbor.set_next_gen(num_of_gen + 1)

    elif prev_neighbor.get_state() == 'S4' and random.random() < P_S4:
        next_neighbor.set_spreader()
        next_neighbor.set_next_gen(num_of_gen + 1)


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


def check_and_update_state(prev_gen, next_gen, i, j, num_of_gen):
    count_spreader_neighbors = 0

    if not prev_gen[i][j]:  # if current cell is not human, no need to do anything
        return

    neighbors = get_neighbors(prev_gen, i, j)
    for neighbor in neighbors:  # count number of spreaders neighbors
        row, col = neighbor
        if next_gen[row][col].is_spreader():
            count_spreader_neighbors += 1

    current_state = prev_gen[i][j].get_state()
    if count_spreader_neighbors >= 2:   # if there are 2 or more neighbors, update the state
        next_gen[i][j].set_state(upgrade_state(current_state))
        next_gen[i][j].state_updated()

    elif prev_gen[i][j].is_state_updated() and prev_gen[i][j].get_next_gen == num_of_gen:
        next_gen[i][j].set_state(return_to_previous_state(current_state))


def spread_rumor(prev_gen_grid, next_gen_grid, i, j, num_of_gen):

    if not prev_gen_grid[i][j]:     # if current cell is not human, no need to do anything
        return

    if not prev_gen_grid[i][j].is_spreader() or prev_gen_grid[i][j].get_next_gen() > num_of_gen:    # if human is spreader but
        # we are not in the num of generation where he can spread again
        return

    neighbors = get_neighbors(prev_gen_grid, i, j)
    for neighbor in neighbors:
        row, col = neighbor
        if not prev_gen_grid[row][col].is_receiver():
            next_gen_grid[row][col].set_receiver()

        decide_if_neighbor_will_spread(prev_gen_grid, next_gen_grid, row, col, num_of_gen)

    next_gen_grid[i][j].set_next_gen(num_of_gen + L)    # current human spreaded the rumor to his neighbors.
    # now we need to update the next generation where he can spread again.


def deep_copy_of_grid(grid):
    copied_grid = [[None] * GRID_SIZE[0] for _ in range(GRID_SIZE[1])]
    for i in range(N):
        for j in range(N):
            current_cell = grid[i][j]
            if not current_cell:
                copied_grid[i][j] = None
                continue
            human_fields = current_cell.get_all_fields()
            copied_grid[i][j] = Human(human_fields[0], human_fields[1], human_fields[2], human_fields[3], human_fields[4])

    return copied_grid


def update_generation(prev_gen, num_of_gen):
    next_gen = deep_copy_of_grid(prev_gen)
    for i in range(N):
        for j in range(N):
            spread_rumor(prev_gen, next_gen, i, j, num_of_gen)  # if current cell is Human, check if spreader
            # and spread according to his neighbors' state
            check_and_update_state(prev_gen, next_gen, i, j, num_of_gen)    # update Human's state if needed
    return next_gen


# choose one Human cell to be the starter of the rumor
def set_starter(grid):

    starter_col = random.randint(0, N - 1)
    starter_row = random.randint(0, N - 1)

    while not grid[starter_row][starter_col]:
        starter_col = random.randint(0, N - 1)
        starter_row = random.randint(0, N - 1)

    first_spreader = grid[starter_row][starter_col]
    first_spreader.set_receiver()
    first_spreader.set_spreader()
    print("the starter is: ", starter_row, starter_col)


def init_grid():
    grid = [[0] * GRID_SIZE[0] for _ in range(GRID_SIZE[1])]
    random.seed(time.time())    # seed(0) if we want to set permanent state

    # Iterate through the grid and set the value of each cell based on the random value and the threshold
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            prob_to_human = random.random()
            if prob_to_human <= P:
                grid[i][j] = Human(random.choice(['S1', 'S2', 'S3', 'S4']))  # init Human object in each cell
                # consider each human as non-believer yet
                # each state is chosen randomly
            else:
                grid[i][j] = None  # not a human

    for i in range(starters_num):
        set_starter(grid)

    return grid


def run_and_animate_generations(grid, state_colors):
    frames = []
    rumur_heared_array = []
    population_counter = 0

    for generation in range(NUM_GENERATIONS):

        # Create an image of the grid using the state colors and add it to the frames list
        image = Image.new('RGB', GRID_SIZE)
        draw = ImageDraw.Draw(image)
        rumur_heared = 0
        for i in range(GRID_SIZE[0]):
            for j in range(GRID_SIZE[1]):
                if grid[i][j]:
                    draw.rectangle((j, i, j, i), fill=state_colors[(grid[i][j].is_spreader(), grid[i][j].get_state())])
                    # color cell with spreader Humans (believers).
                    # if we want to see who received the rumor, change to grid[i][j].is_receiver()
                if (type(grid[i][j]) is Human):
                    if (grid[i][j].is_receiver()):
                        rumur_heared += 1

        rumur_heared_array.append(rumur_heared)
        frames.append(np.array(image))
        # Update the grid
        print(get_states(grid))
        grid = update_generation(grid, generation)

    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            if (type(grid[i][j]) is Human):
                population_counter += 1

    population_rate = rumur_heared_array[NUM_GENERATIONS-1] / population_counter

    return frames, population_rate, rumur_heared_array, population_counter


def get_states(grid):
    states_in_2d = [[0] * GRID_SIZE[0] for _ in range(GRID_SIZE[1])]
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            states_in_2d[i][j] = grid[i][j].get_state() if grid[i][j] else 0
    return states_in_2d


def plot_maker(x_axes, y_axes, color, label_x_axes, label_y_axes, plot_label):
    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x_axes, y_axes, color)
    ax.set_xlabel(label_x_axes)
    ax.set_ylabel(label_y_axes)
    #plt.savefig(plot_label)
    ax.set_title(plot_label)
    plt.show()
    plt.close()

def create_plot(grid, state_colors):
    main_array_10 = []
    for i in range(10):
        frames, population_rate, rumur_heared_array, population_counter = run_and_animate_generations(grid, state_colors)
        main_array_10.append(rumur_heared_array)

    # Average the values in the rumur_heared_arrays list
    avg_rumur_heared_array = np.mean(main_array_10, axis=0)

    avg_rumur_population_array = [member / population_counter for member in avg_rumur_heared_array]


    # create a plot of the population who heared the rumur during all generations
    plot_maker(range(0, NUM_GENERATIONS), avg_rumur_population_array, 'purple', 'number of generations',
               'rumur population rate', 'plot according user choice.png')




def wrap_s1_with_s4(grid):
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            if grid[i][j]:
                if grid[i][j].get_state() == "S1":
                    neighbors = get_neighbors(grid, i, j)
                    for neighbor in neighbors:
                        row = neighbor[0]
                        col = neighbor[1]
                        grid[row][col].set_state("S4")

    return grid


def init_grid_with_more_s3_s4():
    # The weights represent the probabilities of each value being chosen
    # In this case, "s3" and "s4" have twice the probability of "s1" and "s2"
    values = ["S1", "S2", "S3", "S4"]
    weights = [1, 1, 2, 2]

    grid = [[0] * GRID_SIZE[0] for _ in range(GRID_SIZE[1])]
    random.seed(time.time())  # seed(0) if we want to set permanent state

    # Iterate through the grid and set the value of each cell based on the random value and the threshold
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            prob_to_human = random.random()
            if prob_to_human <= P:
                grid[i][j] = Human(random.choices(values, weights)[0])  # init Human object in each cell
                # consider each human as non-believer yet
                # each state is chosen randomly
            else:
                grid[i][j] = None  # not a human

    set_starter(grid)

    return grid


def init_grid_with_s1_far_from_humans():

    grid = [[0] * GRID_SIZE[0] for _ in range(GRID_SIZE[1])]
    random.seed(time.time())  # seed(0) if we want to set permanent state

    # Iterate through the grid and set the value of each cell based on the random value and the threshold
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            prob_to_human = random.random()
            if prob_to_human <= P:
                grid[i][j] = Human("S1")  # init Human object in each cell
                # consider each human as non-believer yet
                # set each one to s1 as default, will be changed soon.
            else:
                grid[i][j] = None  # not a human

    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            if grid[i][j]:
                neighbors = get_neighbors(grid, i, j)
                if len(neighbors) >= 2:
                    # S1 cells will have only one neighbor
                    grid[i][j].set_state(random.choice(["S2", "S3", "S4"]))

    set_starter(grid)

    return grid


def reset_globals():
    global P, L, NUM_GENERATIONS, P_S1, P_S2, P_S3, P_S4, starters_num
    try:
        P = double(input(f"Enter the value for P (default {P}): ") or P)
        L = int(input(f"Enter the value for L (default {L}): ") or L)
        NUM_GENERATIONS = int(input(f"Enter the value for NUM_GENERATIONS (default {NUM_GENERATIONS}): ") or NUM_GENERATIONS)
        P_S1 = double(input(f"Enter the value for P_S1 (default {P_S1}): ") or P_S1)
        P_S2 = double(input(f"Enter the value for P_S2 (default {P_S2}): ") or P_S2)
        P_S3 = double(input(f"Enter the value for P_S3 (default {P_S3}): ") or P_S3)
        P_S4 = double(input(f"Enter the value for P_S4 (default {P_S4}): ") or P_S4)
        starters_num = int(input(f"Enter the value for number of starters peaple (default {starters_num}): ") or starters_num)
    except ValueError:
        print("Invalid input. Please enter an integer value.")
        reset_globals()


def get_input():
    while(1):
        customize = input("Do you want to customize parameter values? (y/n) ")
        if customize.lower() not in ['n', 'y']:
            print("Invalid input, please try again.")
            continue
        else:
            break

    # If user chooses to customize parameter values, prompt for input
    if customize.lower() == "y":
        reset_globals()
        # Prompt the user for input and execute the corresponding function

    while(1):
        model_choice = input("Choose what population do you want to simulate a rumor spreading in:\n"
                       "1. a random population without special conditions\n"
                       "2. a population where every person with S1 state is wrapped with persons with S4 state\n"
                       "3. a population where most of the people are not easy believers (are with S3 and S4 state)\n"
                       "4. a population where people with S1 state are isolated from the other people\n")

        if model_choice not in ['1', '2', '3', '4']:
            print("Invalid choice, please try again.")
            continue

        else:
            if model_choice == '1':
                grid = init_grid()

            if model_choice == '2':
                grid = init_grid()
                wrap_s1_with_s4(grid)

            if model_choice == '3':
                grid = init_grid_with_more_s3_s4()

            if model_choice == '4':
                grid = init_grid_with_s1_far_from_humans()

            break

    return grid


def main():
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

    # Ask user if they want to showing animation or creating plots
    while(1):
        choice = input("for showing the animation choose A OR for creating plots choose P (A/P) ")
        if choice.lower() not in ['p', 'a']:
            print("Invalid input, please try again.")
            continue
        else:
            break

    if (choice.lower() == "a"):
        grid = get_input()

        population_rate_array = []

        # Create a colormap from the state colors dictionary
        cmap = ListedColormap(list(set(state_colors.values())))

        frames, population_rate, rumur_heared_array, population_counter = run_and_animate_generations(grid,
                                                                                                      state_colors)
        population_rate_array.append(population_rate)

        # Create an animation from the frames and display it
        fig = plt.figure(figsize=(8, 8))
        animation = FuncAnimation(fig, lambda i: plt.imshow(frames[i], cmap=cmap), frames=len(frames),
                                  interval=10, repeat=False)
        plt.show()

    # If user chooses to customize parameter values, prompt for input
    if (choice.lower() == "p"):
        grid = get_input()

        create_plot(grid, state_colors)


if __name__ == "__main__":
    main()


