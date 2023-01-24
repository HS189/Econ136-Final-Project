# Green Compute Cluster

## Install Dependencies
Use your favorite dependency manager to install the required dependencies for this project, i.e:
`conda env create -f environment.yml`

## Run Compute Cluster Simulator

To reproduce the results in Sections 4.1 and 4.2 of the paper, run the following commands. While a random seed has been fixed, it is possible that
randomness in your machine causes the generated workload to be slightly different than the one used in the paper. Otherwise, you can run the simulation on the real-life Google workload, which is described in the `google_cluster` folder. <br />
`python3 compute_cluster.py --solar_only [--use-google]` <br />

## Relevant Files

`auction.py` contains implementations of the uniform price auction and the Random Sampling Optimal Price (RSOP) auction. This is used by our dynamic pricing model in the compute cluster. <br /> <br />
`power_supply.py` provides an easy-to-use API that returns available renewable energy on a minute-by-minute granularity. Inputs to methods in this API are timestamps (in seconds, assumed to be multiple of 60) that are mapped to the week-long period the simulation operates on. <br /> <br />
`compute_cluster.py` is the core of the cluster simulation. It generates a workload of 175,000 jobs (as outlined in our paper) or uses the Google workload and then runs these jobs through the compute cluster based on the specified model on green power and the dynamic auction mechanism described in our paper. It collects a variety of useful statistics and results, many of which are presented in our final paper.<br /><br />
`econ.m` and `load_values_2.m` implemented the cubic spline interpolation we developed (see our paper) to convert raw field data (solar intensity, wind speed, etc.) into a realistic power supply (at a minute-level granularity) we could use for powering the compute cluster. Outputs from these methods were used to implement the API in `power_supply.py`. The 2 was used in file and variable names to correspond to the second supporting file to the paper https://doi.org/10.1021/acssuschemeng.0c01054 (year 2013). This was done so that the code could easily be adapted to account for the other files (years 2014-2017).
