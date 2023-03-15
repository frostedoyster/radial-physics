#!/bin/bash
DATASETS=("gold" "qm9" "random-ch4-10k") ## dataset array # CH4 was done by hand due to formatting (calculations were done way before)

for j in ${DATASETS[@]};do # loop over datasets
    echo $j
    for i in 2 14 16 17 18 19 20 21 22 23 24 25 26 27
    do
        for e in 300 400 500 600 700;
        do
            grep 'factor' TR$i"_"*E$e'_'*$j.out | awk '{print $3}' > Factors_RT$i'_'$j ## extract factors
            grep 'E_max_2' TR$i"_"*E$e'_'*$j.out | awk '{print $3}' > Emax_RT$i'_'$j ## extract emax
            grep 'Test MAE' TR$i"_"*E$e'_'*$j.out | awk '{print $6}' | sed 's/.$//' > Test_MAE_RT$i'_'$j ## extract MAE
            pr -mts' ' Factors_RT$i'_'$j Emax_RT$i'_'$j > TMP_MAE_Data_TR$i'_'$j ## combine files horizontally, write to new file
            pr -mts' ' TMP_MAE_Data_TR$i'_'$j Test_MAE_RT$i'_'$j > MAE_Data_TR$i'_'E$e'_'$j

            rm Factors_RT$i'_'$j
            rm Test_MAE_RT$i'_'$j
            rm Emax_RT$i'_'$j
            rm TMP_MAE_Data_TR$i'_'$j
            sort -n -o MAE_Data_TR$i'_'E$e'_'$j MAE_Data_TR$i'_'E$e'_'$j # propertly sort from lowest to highest factor
        done
    done
done
