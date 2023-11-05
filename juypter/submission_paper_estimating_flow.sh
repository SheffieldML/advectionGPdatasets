#!/bin/bash
# Request 100Gb of real memory (mem)
#SBATCH --mem=150G
#SBATCH --cpus-per-task=10

# load the module for the program we want to run
#module load apps/gcc/foo

/users/md1xmtsx/anaconda3/bin/python paper_estimating_flow3d.py
