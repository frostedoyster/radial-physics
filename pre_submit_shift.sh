#!/bin/bash

#0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4
#0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
#0.2 0.25 0.3 0.4 0.5
for j in 0.0 0.1 0.2 0.3 0.4 #shift
do
    for i in 1.4 1.6 1.8 2.0 2.2 #factor
    do
        sbatch submit_job_shift.sh gpr_LE.py $i $j
        echo "Factor $i Shift $j"
        sleep 0.3
    done
done