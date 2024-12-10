This repository contains working code for paper **'On the Complexity of Teaching a Family of Linear Behavior Cloning Learners'** accepted at NeurIPS 2024. We provide steps for running all the three environments used in the paper below.


# Steps to run the code :

## Create conda environment and install dependencies

- conda env create --name teach_bc -f env.py


## Running the code

- run : python driver.py.
- to save output set 'record' parameter in params.yml file to True.
- you can also set how detailed output results you want to save using 'save_output' parameters.

Note that when running with record=False the output is still saved to no_record directory but the outputs are not persistent and is overwritten next time the code is run with record=False.


# Examples and Parameters :

Common parameters :

- project_dir : the main project directory in which code and outputs directory is present.
- output_dir : the directory where outputs should be saved.
- terminal_output_to_log_file : turn this to True to save terminal output to a log file.
- instance_type : used to set one of the three different environment types we want to run. consequently, the environment specific parameters should be updated as detailed below.
- save_output : specify how detailed output to be saved on running an experiment.
- record : specify whether to save output or not.
- run_parallel : use it to run multiple instance of a particular environment type. update the corresponding parameters for each environment in parallel parameters list.
- num_jobs_in_parallel : total number of threads to use to execute parallel jobs.
- num_randomized_trials : number of trials Teach-Random algorithm should be run.

## Pick the right diamond example 
diamond_edges : set list of different number of diamond edges available.
num_slots : set number of slots in the board.


## Polygon tower example
- s_being : set starting state numner.
- s_end : set ending state number.


## Visual programming example :
grid_size_list : set the size $n$ of the maze grid environment.
repeat_moves : specify whether to allow agent to use repeat actions; we set to True for our experiments.
feature_type : specify the type of features - global or local to be used.




# Plotting 

- Code to plot the figures in paper can be found in plot_graphs.ipynb notebook file.

# Computational Resources

All the experiments in this project has been run on a Macbook Pro M1 system with 16GB of RAM and 512GB GB of SSD storage.


