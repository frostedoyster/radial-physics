#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 28
#SBATCH --mem 120G
#SBATCH --time 2-0

##########################################
##################INPUT:##################  sbatch ...
##########################################

############## (HYPER)PARAMETERS ##############
CUTOFF_RADIUS=4.5
RAD_TR_SELECTION=1
RAD_TR_FACTOR=$2 # pass as command line argument
RAD_TR_DISPLACEMENT=0
DATASET_PATH='datasets/random-ch4-10k.extxyz' # 'datasets/random-ch4-10k.extxyz' 'datasets/gold.xyz' 'datasets/qm9.xyz'
DATASET_TARGET_KEY='energy' # 'energy' 'elec._Free_Energy_[eV]' 'U0'
N_TRAIN=1000
N_TEST=1000
################################################

TEMP_VAR=${DATASET_PATH:9:-1}
DATA_SET=${TEMP_VAR//.*} # just some string manipulations

# AUTOMATIC OUTPUT FILE NAME:
OUTPUT_NAME='TR'$RAD_TR_SELECTION'_F'$RAD_TR_FACTOR'_D'$RAD_TR_DISPLACEMENT'_'$N_TRAIN'_'$N_TEST'_a'$CUTOFF_RADIUS'_'$DATA_SET.out
OUTPUT_PATH='/home/ach/radial-physics_new/radial-physics/output/tests'

# module load gcc
# module load python
# source /home/ach/fidis_venv/bin/activate

export OMP_NUM_THREADS=28
echo STARTING AT `date` > $OUTPUT_PATH/$OUTPUT_NAME
python -u $1 $CUTOFF_RADIUS $RAD_TR_SELECTION $RAD_TR_FACTOR $DATASET_PATH $DATASET_TARGET_KEY $N_TRAIN $N_TEST $RAD_TR_DISPLACEMENT\
  >> $OUTPUT_PATH/$OUTPUT_NAME
echo FINISHED at `date` >> $OUTPUT_PATH/$OUTPUT_NAME
#rm splines/*
#rm slurm*
#mv slurm-* /home/ach/radial-physics_new/radial-physics/slurms