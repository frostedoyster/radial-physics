#!/bin/bash
DATASETS=("gold" "qm9" "random-ch4-10k") ## dataset array

for j in ${DATASETS[@]};do # loop over datasets
    echo $j
    for i in 0
    do
        grep 'E_max_2' TR$i"_"*_$j.out | awk '{print $3}' > Emax_RT$i'_'$j ## extract emax
        grep 'Cutoff Radius' TR$i"_"*_$j.out | awk '{print $4}' > Cutoff_RT$i'_'$j ## extract a
        grep 'n_test' TR$i"_"*_$j.out | awk '{print $3}' > n_test$i'_'$j ## extract n
        grep 'Test RMSE' TR$i"_"*_$j.out | awk '{print $3}' > Test_RMSE_RT$i'_'$j ## extract RMSE
        pr -mts'_' Cutoff_RT$i'_'$j Emax_RT$i'_'$j > TMP_Data_TR$i'_'$j
        pr -mts'_' TMP_Data_TR$i'_'$j n_test$i'_'$j > TMP2_Data_TR$i'_'$j
        pr -mts' ' TMP2_Data_TR$i'_'$j Test_RMSE_RT$i'_'$j > Data_TR$i'_'$j ## combine two files horizontally, write to new file

        rm TMP_Data_TR$i'_'$j
        rm TMP2_Data_TR$i'_'$j
        rm Cutoff_RT$i'_'$j
        rm Emax_RT$i'_'$j
        rm n_test$i'_'$j
        rm Test_RMSE_RT$i'_'$j
        #rm Factors_RT$i'_'$j
        #rm Test_RMSE_RT$i'_'$j
        #sort -n -o Data_TR$i'_'$j Data_TR$i'_'$j # propertly sort from lowest to highest factor
    done
done