#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 28
#SBATCH --mem 180G
#SBATCH --time 0-1

##########################################
##################INPUT:##################  sbatch ...
##########################################

############## (HYPER)PARAMETERS ##############
CUTOFF_RADIUS=6.0
RAD_TR_SELECTION=$3
RAD_TR_FACTOR=$2 # pass as command line argument
RAD_TR_DISPLACEMENT=0
DATASET_PATH='datasets/random-ch4-10k.extxyz' # 'datasets/random-ch4-10k.extxyz' 'datasets/gold.xyz' 'datasets/qm9.xyz'
N_TRAIN_TEST=9000
E_MAX_2=$4 #400
################################################

TEMP_VAR=${DATASET_PATH:9:-1}
DATA_SET=${TEMP_VAR//.*} # just some string manipulations

# AUTOMATIC OUTPUT FILE NAME:
OUTPUT_NAME='TR'$RAD_TR_SELECTION'_F'$RAD_TR_FACTOR'_D'$RAD_TR_DISPLACEMENT'_E'$E_MAX_2'_'$N_TRAIN_TEST'_a'$CUTOFF_RADIUS'_'$DATA_SET.out
OUTPUT_PATH='/home/ach/radial-physics_new/radial-physics/output/All_trafo_all_n/10000_new' #'/home/ach/radial-physics_new/radial-physics/output/TR2_variants/a_'$CUTOFF_RADIUS'_N_'$N_TRAIN_TEST'_E_'$E_MAX_2

# module load gcc
# module load python
# source /home/ach/fidis_venv/bin/activate

export OMP_NUM_THREADS=28
echo STARTING AT `date` > $OUTPUT_PATH/$OUTPUT_NAME
python -u $1 $CUTOFF_RADIUS $RAD_TR_SELECTION $RAD_TR_FACTOR $DATASET_PATH $N_TRAIN_TEST $E_MAX_2 $RAD_TR_DISPLACEMENT\
  >> $OUTPUT_PATH/$OUTPUT_NAME
echo FINISHED at `date` >> $OUTPUT_PATH/$OUTPUT_NAME
#rm splines/*
#rm slurm*
#mv slurm-* /home/ach/radial-physics_new/radial-physics/slurms