#!/bin/bash

#0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4
#0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 3.0 3.5 4.0
for i in 0.6 0.3 0.4 0.5
do
    sbatch submit_job_bpnn.sh bpnn_LE.py $i
    echo "Factor $i"
    sleep 0.3
done