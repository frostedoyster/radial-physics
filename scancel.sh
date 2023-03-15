#!/bin/bash

for j in `seq 187640 1 187645`;do # list: first job, step/increment, last job
    scancel $j
    echo Deleted job no. $j
done
