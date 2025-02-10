#!/bin/bash

# Script for running the algorithm 10 times and calculating the average time. Exercice 1 and 2.

#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 00:05:00
#SBATCH --mem 64G
#SBATCH -J P1_1
#SBATCH -o ./outs/P1.o
#SBATCH -e ./outs/P1.e

module load gcc

gcc -O2 -Wall -lm -o main $1

sum_time=0
max_iter=10

for i in $(seq 1 $max_iter)
do
    output=$(./main)
    time=$(echo "$output" | grep "PAE | Time:" | awk '{print $4}')
    echo "PAE | Code: $1 | Execution $i: $time seconds"
    sum_time=$(echo "$sum_time + $time" | bc)
done

avg_time=$(echo "scale=6; $sum_time / $max_iter" | bc)
echo "PAE | Code: $1 | Total: $sum_time | Average: $avg_time seconds"

rm main
