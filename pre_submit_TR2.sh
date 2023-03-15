#!/bin/bash

#0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4

for i in 1.7 1.9 3.1 3.3 #0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 4.8 5.0 5.2 5.4 5.6 5.8 6.0
do
    sbatch sj_TR2.sh gpr_LE.py $i
    #sbatch submit_job.sh gpr_LE_scal.py $i
    echo "Factor $i"
    sleep 0.1
done