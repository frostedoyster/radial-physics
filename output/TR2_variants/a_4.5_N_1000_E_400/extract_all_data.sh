#!/bin/bash
DATASETS=("gold" "qm9" "random-ch4-10k") ## dataset array

for j in ${DATASETS[@]};do # loop over datasets
    echo $j
    for i in 2 14 15 16 17 18 19 20 21 22 23 24 26 27 # 25: current Data file is good, don't use this script on it
    do
        grep 'factor' TR$i"_"*_$j.out | awk '{print $3}' > Factors_RT$i'_'$j ## extract factors
        grep 'Test RMSE' TR$i"_"*_$j.out | awk '{print $3}' > Test_RMSE_RT$i'_'$j ## extract RMSE
        pr -mts' ' Factors_RT$i'_'$j Test_RMSE_RT$i'_'$j > Data_TR$i'_'$j ## combine two files horizontally, write to new file

        rm Factors_RT$i'_'$j
        rm Test_RMSE_RT$i'_'$j
        sort -n -o Data_TR$i'_'$j Data_TR$i'_'$j # propertly sort from lowest to highest factor
    done
done
