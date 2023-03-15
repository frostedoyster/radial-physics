#!/bin/bash
DATASETS=("gold" "qm9" "random-ch4-10k") ## dataset array # CH4 was done by hand due to formatting (calculations were done way before)

for j in ${DATASETS[@]};do # loop over datasets
    echo $j
    for i in 0
    do
        for e in 300 400 500 600 700;
        do
            grep 'E_max_2' TR$i"_"*E$e'_'*$j.out | awk '{print $3}' ## print emax
            grep 'Test RMSE' TR$i"_"*E$e'_'*$j.out | awk '{print $3}' ## print RMSE
        done
    done
done
