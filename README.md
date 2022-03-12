# Econ136-Final-Project

## Install Dependencies
Use your favorite dependency manager to install the required dependencies for this project, i.e:
`conda env create -f environment.yml`

## Run Compute Cluster Simulator

To reproduce the results in Sections 4.1 and 4.2 of the paper, run the following commands. While a random seed has been fixed, it is possible that
randomness in your machine causes the generated workload to be slightly different than the one used in the paper.<br />
`python3 compute_cluster.py --solar_only` <br />
`python3 compute_cluster.py --combined`

## Relevant Files

`auction.py` contains implementations of the uniform price auction and the Random Sampling Optimal Price (RSOP) auction. This is used by our dynamic pricing model in the compute cluster. <br />
`power_supply.py` provides an easy-to-use API that returns available renewable energy on a minute-by-minute granularity. Inputs to methods in this API are timestamps (in seconds, assumed to be multiple of 60) that are mapped to the week-long period the simulation operates on. <br />
`compute_cluster.py` is the core of the cluster simulation. It generates a workload of 175,000 jobs (as outlined in our paper) and then runs these jobs through the compute cluster based on the specified model on green power and the dynamic auction mechanism described in our paper. It collects a variety of useful statistics and results, many of which are presented in our final paper.
