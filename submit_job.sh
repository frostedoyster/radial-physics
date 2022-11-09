#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 28
#SBATCH --mem 120G
#SBATCH --time 2-0
CUTOFF_RADIUS=4.5
RAD_TR_SELECTION=1
RAD_TR_FACTOR=2.0
export OMP_NUM_THREADS=28
echo STARTING AT `date`
python -u $1 $CUTOFF_RADIUS $RAD_TR_SELECTION $RAD_TR_FACTOR > output/something.out
echo FINISHED at `date`

