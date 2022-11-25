#!/bin/bash
DATASETS=("random-ch4-10k") ## dataset array # "gold" "qm9"

for j in ${DATASETS[@]};do # loop over datasets
    echo $j
    for i in 1 2 3 4 5 6 7 8
    do
        grep 'factor' TR$i"000_"*_$j.out | awk '{print $3}' > Factors_RT$i'000_'$j ## extract factors
        grep 'Test RMSE' TR$i"000_"*_$j.out | awk '{print $3}' > Test_RMSE_RT$i'000_'$j ## extract RMSE
        pr -mts' ' Factors_RT$i'000_'$j Test_RMSE_RT$i'000_'$j > Data_TR$i'000_'$j ## combine two files horizontally, write to new file

        rm Factors_RT$i'000_'$j
        rm Test_RMSE_RT$i'000_'$j
        sort -n -o Data_TR$i'000_'$j Data_TR$i'000_'$j # propertly sort from lowest to highest factor
    done
done