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
        grep 'displacement' TR$i"_"*_$j.out | awk '{print $3}' > Displacement_RT$i'_'$j ## extract displacement
        pr -mts'_' Cutoff_RT$i'_'$j Emax_RT$i'_'$j > TMP_Data_TR$i'_'$j
        pr -mts'_' TMP_Data_TR$i'_'$j n_test$i'_'$j > TMP2_Data_TR$i'_'$j
        pr -mts' ' TMP2_Data_TR$i'_'$j Displacement_RT$i'_'$j > TMP3_Data_TR$i'_'$j
        pr -mts' ' TMP3_Data_TR$i'_'$j Test_RMSE_RT$i'_'$j > Data_TR$i'_'$j ## combine two files horizontally, write to new file

        rm Displacement_RT$i'_'$j
        rm TMP_Data_TR$i'_'$j
        rm TMP2_Data_TR$i'_'$j
        rm TMP3_Data_TR$i'_'$j
        rm Cutoff_RT$i'_'$j
        rm Emax_RT$i'_'$j
        rm n_test$i'_'$j
        rm Test_RMSE_RT$i'_'$j
        #rm Factors_RT$i'_'$j
        #rm Test_RMSE_RT$i'_'$j
        #sort -n -o Data_TR$i'_'$j Data_TR$i'_'$j # propertly sort from lowest to highest factor
    done
        grep '4.5_400_1000' Data_TR0_$j > Data_TR0_$j'_'4.5_400_1000
        grep '6.0_400_1000' Data_TR0_$j > Data_TR0_$j'_'6.0_400_1000
        grep '6.0_400_2000' Data_TR0_$j > Data_TR0_$j'_'6.0_400_2000
        grep '6.0_400_500' Data_TR0_$j > Data_TR0_$j'_'6.0_400_500
        grep '6.0_800_1000' Data_TR0_$j > Data_TR0_$j'_'6.0_800_1000
done