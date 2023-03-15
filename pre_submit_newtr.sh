#!/bin/bash

#0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4
#0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
#0.05 0.1 0.15 0.2 0.25 0.3 0.4
#0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
#TRANSFORM=2
################################################################################################
for TRANSFORM in 0
do
    for i in 1.0 #0.2 0.3 0.25 0.15 0.35 0.45 0.4 0.5 0.6 0.7 0.8 # 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.2     
    do
        for EMAX in 300 400 500 600 700
        do
        sbatch sj_newtr.sh gpr_LE.py $i $TRANSFORM $EMAX #####rm emax later # factor and transform selection
        echo "Factor $i Tr $TRANSFORM $EMAX" #####rm emax later
        #sleep 0.1
        done
    done
done

# for TRANSFORM in 2 14 16 17 18 19 20 21 22 23 24 25 26 27 #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
# do
#     for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.8 3.0 3.5 4.0
#     do
#         for EMAX in 300 400 500 600 700
#         do
#             sbatch sj_newtr.sh gpr_LE.py $i $TRANSFORM $EMAX #####rm emax later # factor and transform selection
#             echo "Factor $i Tr $TRANSFORM $EMAX" #####rm emax later
#             #sleep 0.1
#         done
#     done
# done

# ###QM9
# for TRANSFORM in 2 14 21 23 24 22 #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
# do
#     for i in 0.9 1.1 1.0 1.2 1.3 1.4 1.5 1.6 1.7 1.8 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.8 3.0 3.5 4.0
#     do
#         for EMAX in 300 400 500 600 700
#         do
#             sbatch sj_newtr.sh gpr_LE.py $i $TRANSFORM $EMAX #####rm emax later # factor and transform selection
#             echo "Factor $i Tr $TRANSFORM $EMAX" #####rm emax later
#             #sleep 0.1
#         done
#     done
# done

# for TRANSFORM in 19 20 25 26 27 16 17 18  #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
# do
#     for i in 1.0 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.8 3.0 3.5 4.0
#     do
#         for EMAX in 300 400 500 600 700
#         do
#             sbatch sj_newtr.sh gpr_LE.py $i $TRANSFORM $EMAX #####rm emax later # factor and transform selection
#             echo "Factor $i Tr $TRANSFORM $EMAX" #####rm emax later
#             #sleep 0.1
#         done
#     done
# done

###CH4
for TRANSFORM in 2 26 27 #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    for i in 0.1 0.2 0.3 0.4 0.5 0.6 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.8 3.0 3.5 4.0
    do
        for EMAX in 300 400 500 600 700
        do
            sbatch sj_newtr.sh gpr_LE.py $i $TRANSFORM $EMAX #####rm emax later # factor and transform selection
            echo "Factor $i Tr $TRANSFORM $EMAX" #####rm emax later
            #sleep 0.1
        done
    done
done

for TRANSFORM in 16 25 22 14 19 20 #19 20 25 22 14 16 17 18 21 23 24  #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    for i in 1.0 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.8 3.0 3.5 4.0
    do
        for EMAX in 300 400 500 600 700
        do
            sbatch sj_newtr.sh gpr_LE.py $i $TRANSFORM $EMAX #####rm emax later # factor and transform selection
            echo "Factor $i Tr $TRANSFORM $EMAX" #####rm emax later
            #sleep 0.1
        done
    done
done

# ###GOLD
# for TRANSFORM in 2 14 16 17 18 21 23 24 #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
# do
#     for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.8 3.0 3.5 4.0
#     do
#         for EMAX in 300 400 500 600 700
#         do
#             sbatch sj_newtr.sh gpr_LE.py $i $TRANSFORM $EMAX #####rm emax later # factor and transform selection
#             echo "Factor $i Tr $TRANSFORM $EMAX" #####rm emax later
#             #sleep 0.1
#         done
#     done
# done

# for TRANSFORM in 19 20 25 22 26 27  #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
# do
#     for i in 1.0 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.8 3.0 3.5 4.0
#     do
#         for EMAX in 300 400 500 600 700
#         do
#             sbatch sj_newtr.sh gpr_LE.py $i $TRANSFORM $EMAX #####rm emax later # factor and transform selection
#             echo "Factor $i Tr $TRANSFORM $EMAX" #####rm emax later
#             #sleep 0.1
#         done
#     done
# done
################################################################################################
