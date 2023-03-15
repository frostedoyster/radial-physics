#!/bin/bash
DATASETS=("gold" "qm9" "random-ch4-10k") ## dataset array # CH4 was done by hand due to formatting (calculations were done way before)

for j in ${DATASETS[@]};do # loop over datasets
    echo $j
    for i in 2222 #1 2 3 4 5 6 7 8
    do
        grep 'factor' TR$i"_"*_$j.out | awk '{print $3}' > Factors_RT$i'_'$j ## extract factors
        grep 'Test RMSE' TR$i"_"*_$j.out | awk '{print $3}' > Test_RMSE_RT$i'_'$j ## extract RMSE
        pr -mts' ' Factors_RT$i'_'$j Test_RMSE_RT$i'_'$j > Data_TR$i'_'$j ## combine two files horizontally, write to new file

        rm Factors_RT$i'_'$j
        rm Test_RMSE_RT$i'_'$j
        sort -n -o Data_TR$i'_'$j Data_TR$i'_'$j # propertly sort from lowest to highest factor
    done
done