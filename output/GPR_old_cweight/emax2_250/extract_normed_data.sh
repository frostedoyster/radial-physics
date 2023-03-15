#!/bin/bash
DATASETS=("random-ch4-10k") ## dataset array "gold" "qm9"

for j in ${DATASETS[@]};do # loop over datasets
    echo $j
    for i in 1000 5000 6000 # transforms which were already normed (helps jupyer notebook)
    do
        echo nan nan > Data_TR$i'_'$j
    done
    
    for i in 2000 3000 4000 7000 8000
    do
        grep 'factor' TR$i"_"*_$j.out | awk '{print $3}' > Factors_RT$i'_'$j ## extract factors
        grep 'Test RMSE' TR$i"_"*_$j.out | awk '{print $3}' > Test_RMSE_RT$i'_'$j ## extract RMSE
        pr -mts' ' Factors_RT$i'_'$j Test_RMSE_RT$i'_'$j > Data_TR$i'_'$j ## combine two files horizontally, write to new file

        rm Factors_RT$i'_'$j
        rm Test_RMSE_RT$i'_'$j
        sort -n -o Data_TR$i'_'$j Data_TR$i'_'$j # propertly sort from lowest to highest factor
    done
done