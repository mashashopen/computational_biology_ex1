# computational_biology_ex1

this program is an implementation of a rumor spreading model. A population of people is represented as a grid of size 100 * 100, with parameters as:  
P- population density.  
P_Si- probability of a person with level of skepticism Si to beleive the rumor.  
L- number of generations that a person have to wait since spreading the rumor until next time he can pass it again to the neighbors.  
NUM_GENERATIONS- number of generations the model will run.  

one person is chosen randomly to start the rumor. then, the person spreads the rumor to all the neighbors (3 to 8 neighbors, depends on the location on the grid). each neighbor is now a receiver of the rumor, but each will be a spreader only after a calculation of probabilty (i.e person with level of skepticism S2 will spread the rumor with a probabilty of 2/3). We assume that if the person spreads the rumor, he is also a believer (except for the starter who is not necessarily a believer).  

for each person in the grid we check if he is a spreader and if so, we are doing for him the described above.  
one iterations on the whole grid, counts as one generation.  
we continue with the iteration and we can plot the data to see the rate of the rumor spreading, or to vizualize the spreading itself using animation.  


################### How to run ###################   
in order to run this program, please run first:  
pip install -r requirements.txt  

then, run:  
rumor_spreading_model.py   

a menu will be displayed. you can choose to run the simulation with a random grid, or with an initial grid with special conditions.   
then, you can choose to view the animation of the rumor spreading (where the color purple represents the people who spreaded the rumor),   
or you can choose to display the plots of the data.   
