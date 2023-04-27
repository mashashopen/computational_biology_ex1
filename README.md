# computational_biology_ex1

this program is an implementation of a rumor spreading model. A population of people is represented as a grid of size 100 * 100, with parameters as:
P- population density
P_Si- probability of a person with level of skepticism Si to beleive the rumor
L- number of generations that a person have to wait since spreading the rumor until next time he can pass it again to the neighbors
NUM_GENERATIONS- number of generations the model will run


################### How to run ###################
in order to run this program, please run first:
pip install -r requirements.txt

then, run:
rumor_spreading_model.py

a menu will be displayed. you can choose to run the simulation with a random grid, or with an initial grid with special conditions.
then, you can choose to view the animation of the rumor spreading (where the color purple represents the people who spreaded the rumor),
or you can choose to display the plots of the data.
