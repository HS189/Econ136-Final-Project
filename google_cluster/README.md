# Using the Google cluster data dataset

## Background
See https://github.com/google/cluster-data for details regarding this dataset. We used version 2 (https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md).

## Downloading the Data

Install the `gsutil` command line tool (Google provides documentation online on how to install gsutil). Next, run
```
gsutil -m cp -R gs://clusterdata-2011-2/task_events/ task_events/
``` 
to copy the task_events table of the dataset to a local folder called task_events. You should also download the schema.csv file from the bucket. See [v2.1 format + schema document](https://drive.google.com/file/d/0B5g07T_gRDg9Z0lsSTEtTWtpOW8/view?usp=sharing&resourcekey=0-cozD56gA4fUDdrkHnLJSrQ) for more details on the data structure and how to use gsutil.

## Preprocessing

Study and run the code in the `workload.ipynb` notebook to see how we adapt this dataset to a usable format for our compute cluster simulations. When run, the final output should be a file called `all_tasks.csv`.
