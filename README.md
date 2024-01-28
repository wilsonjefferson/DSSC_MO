# DSSC: Mathematical Optimization

In this repo you will find information on Mathematical Optimization course attended at the University of Trieste. Especially, this is the project folder containing an implementation of the scientific paper: "Formulation and solution technique for agricultural waste collection". The project is explained in the PDF presentation.

The paper was pubblished on the European Journal of Operational Research in the 2021.

The aim of the paper is to present a mixed-integer nonlinear programming model for agricultural waste collection and transport network design that aims to stop burning waste and use the waste to produce bio-organic fertilizer. The model supports rural planners to optimally locate waste storages, and to determine the optimal set of routes for a fleet of vehicles to collect and transport the waste from the storages to the bio-organic fertilizer production facility.

In addition, a parallel water flow algorithm is developed to solve efficiently the large-sized instances.

## Structure of the Git Folder

The project folder is organized in several directories, following a short explanation of each of them:

- Folder src - it contains the source code of the project such as the LARP model and a set of support functionalities;

- Folder data - it contains the data shared by the authors of the paper to replicate the results;

- Folder images - it contains the images generated;

- Folder backup - it contains temporary data useful for the execution of the project;

- larp_waste_management.ipynb - Notebook to illustrate each component of the LARP model, such as decision variables, objective function and constrains. This notebook replicate the strategy proposed in the paper and it produce the same results.

- scalability.py - main python script to start the scalability analysis;

- environment.yml - yml file containing the requirements of the project.

## How to run

Once you have installed the requirements defined in the yaml file, you may want to execute the code proposed in larp_waste_management.ipynb notebook to replicate the same results proposed in the scientific paper.

Alternatively, you can execute the scalability.py file to start the Scalability Analysis process. The aim is to evaluate the capability, of the proposed model, to scale for different problem sizes. This means, understand if the model is still able to provide an optimal solution in reasonable time.