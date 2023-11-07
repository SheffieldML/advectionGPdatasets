#!/bin/bash
#SBATCH --mem=220G
#SBATCH --cpus-per-task=20
#
# load the module for the program we want to run
#module load apps/gcc/foo
#/users/md1xmtsx/anaconda3/bin/python paper_estimating_flow3d.py --rep=$SLURM_ARRAY_TASK_ID --features=50000 --particles=50 --ls=2 --k0=0.459 --resT 200
/users/md1xmtsx/anaconda3/bin/python paper_estimating_flow3d.py --rep=$SLURM_ARRAY_TASK_ID --features=100000 --particles=50 --ls=2 --k0=0.459 --resT 200


#test (quick!)
#/users/md1xmtsx/anaconda3/bin/python paper_estimating_flow3d.py --rep=$SLURM_ARRAY_TASK_ID --features=2000 --particles=20 --ls=2 --k0=0.459 --resT 200
