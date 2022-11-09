#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 28
#SBATCH --mem 120G
#SBATCH --time 2-0

############## (HYPER)PARAMETERS ##############
CUTOFF_RADIUS=4.5
RAD_TR_SELECTION=1
RAD_TR_FACTOR=2.0
DATASET_PATH='datasets/random-ch4-10k.extxyz' # 'datasets/random-ch4-10k.extxyz' 'datasets/gold.xyz' 'datasets/qm9.xyz'
DATASET_TARGET_KEY='energy' # 'energy' 'elec. Free Energy [eV]' 'U0'
N_TRAIN=100
N_TEST=100
################################################

TEMP_VAR=${DATASET_PATH:9:-1}
DATA_SET=${TEMP_VAR//.*}

# AUTOMATIC OUTPUT FILE NAME:
OUTPUT_NAME='TR'$RAD_TR_SELECTION'_'$RAD_TR_FACTOR'_'$N_TRAIN'_'$N_TEST'_'$CUTOFF_RADIUS'_'$DATA_SET.out

export OMP_NUM_THREADS=28
echo STARTING AT `date` > output/$OUTPUT_NAME
python -u $1 $CUTOFF_RADIUS $RAD_TR_SELECTION $RAD_TR_FACTOR $DATASET_PATH $DATASET_TARGET_KEY $N_TRAIN $N_TEST \
  >> output/$OUTPUT_NAME
echo FINISHED at `date` >> output/$OUTPUT_NAME