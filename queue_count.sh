#!/bin/bash

squeue -u ach -h -t pending,running -r | wc -l
squeue -u ach -h -t running -r | wc -l; echo `date`
