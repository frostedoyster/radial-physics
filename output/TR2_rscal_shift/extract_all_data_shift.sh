#!/bin/bash
DATASETS=("random-ch4-10k" "gold" "qm9") ## dataset array "gold" "qm9"

for j in ${DATASETS[@]};do # loop over datasets
    echo $j
    for i in 1 2 3 4 5 6 7 8
    do
        grep 'factor' TR$i"_"*_$j.out | awk '{print $3}' > Factors_RT$i'_'$j ## extract factors
        grep 'displacement' TR$i"_"*_$j.out | awk '{print $3}' > Displacement_RT$i'_'$j ## extract displacement
        grep 'Test RMSE' TR$i"_"*_$j.out | awk '{print $3}' > Test_RMSE_RT$i'_'$j ## extract RMSE
        pr -mts' ' Displacement_RT$i'_'$j Test_RMSE_RT$i'_'$j > Temp_Data_TR$i'_'$j ## combine two files horizontally, write to new file
        pr -mts' ' Factors_RT$i'_'$j Temp_Data_TR$i'_'$j > Data_TR$i'_'$j
        rm Factors_RT$i'_'$j
        rm Test_RMSE_RT$i'_'$j
        rm Displacement_RT$i'_'$j
        rm Temp_Data_TR$i'_'$j
        sort -n -o Data_TR$i'_'$j Data_TR$i'_'$j # propertly sort from lowest to highest factor
    done
done