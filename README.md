# Econ136-Final-Project

## Install Dependencies
Use your favorite dependency manager to install the required dependencies for this project, i.e:
`conda env create -f environment.yml`

## Run Compute Cluster Simulator

To reproduce the results in Sections 4.1 and 4.2 of the paper, run the following commands. While a random seed has been fixed, it is possible that
randomness in your machine causes the generated workload to be slightly different than the one used in the paper.<br />
`python3 compute_cluster.py --solar_only` <br />
`python3 compute_cluster.py --combined`
