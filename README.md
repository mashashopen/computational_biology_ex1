# computational_biology_ex1

this program is an implementation of a rumor spreading model. A population of people is represented as a grid of size 100 * 100, with parameters as:  
P- population density.  
P_Si- probability of a person with level of skepticism Si to beleive the rumor.  
L- number of generations that a person have to wait since spreading the rumor until next time he can pass it again to the neighbors.  
NUM_GENERATIONS- number of generations the model will run.  

One person is chosen randomly to start the rumor. Then, the person spreads the rumor to all the neighbors (3 to 8 neighbors, depends on the location on the grid). Each neighbor is now a receiver of the rumor, but each will be a spreader only after a calculation of probabilty (i.e person with level of skepticism S2 will spread the rumor with a probabilty of 2/3). We assume that if the person spreads the rumor, he is also a believer (except for the starter who is not necessarily a believer).  

For each person in the grid we check if he is a spreader and if so, we are doing for him the described above.  
One iteration on the whole grid, counts as one generation.  
We continue with the iterations and we can plot the data to see the rate of the rumor spreading, or to vizualize the spreading itself using animation.

Note, that the plots are calculating number of people who *recieved* the rumor whereas the animation colors the people who *spreaded* the rumor (a.k.a, believers).  


################### How to run ###################   
The executable file can be found here: https://drive.google.com/drive/u/0/folders/128sB7iPpJALXjwNSs87UKYVNx0ccdwgG  

Please make sure to wait for the first output (a question for the user) before inserting any input).  
##################################################  
    
Before viewing the results, you will have to answer a few questions.  
First, you will choose if you want to view the animation of the rumor spreading (where the color purple represents the people who spreaded the rumor),   
or to display the plots of the data.  
Then, you will choose if you want to set your own parameters (descrived on the first paragraph).  
Last, you will choose to run the simulation with a random grid, or with an initial grid with special conditions.

Afterwards the result will be displayed.
