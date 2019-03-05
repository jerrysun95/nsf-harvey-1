#!/bin/bash
#-----------------------------------------------------------------
# Example SLURM job script to run serial applications on TACC's
# Stampede system.
#
# This script requests one core (out of 16) on one node. The job
# will have access to all the memory in the node.  Note that this
# job will be charged as if all 16 cores were requested.
#-----------------------------------------------------------------

#SBATCH -J jaccard           # Job name
#SBATCH -o jaccard%j.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p normal           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 10:00:00              # Run time (hh:mm:ss) - 1.5 hours


# Load any necessary modules (these are examples)
# Loading modules in the script ensures a consistent environment.

date
source activate nsf
python find_num_topics.py
