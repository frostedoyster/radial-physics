#!/bin/bash

#0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4
#0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
#0.05 0.1 0.15 0.2 0.25 0.3 0.4

for TRANSFORM in 28 #2 14 15 16 17 18 19 20 21 22 23 24 25 26 27 #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.8 3.0 3.5 4.0
    do

        sbatch sj_2Var.sh gpr_LE.py $i $TRANSFORM # factor and transform selection
        echo "Factor $i Tr $TRANSFORM"
        #sleep 0.1

    done
done