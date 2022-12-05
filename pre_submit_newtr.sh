#!/bin/bash

#0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4
#0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
for i in 0.2 0.4 0.6 0.8 1.0 1.2
do
    sbatch submit_job_newtr.sh gpr_LE.py $i
    echo "Factor $i"
    sleep 0.3
done