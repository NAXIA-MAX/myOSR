#!/bin/bash

num_cv=10

for num_unknown in 1 2 3 4 5
do
    echo "Running with num_unknown=$num_unknown, num_cv=$num_cv"
    python runCVtest.py $num_unknown $num_cv
done
