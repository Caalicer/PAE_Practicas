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

total_exec_time=0
total_memory_time=0

max_iter=10

for i in $(seq 1 $max_iter)
do

    output=$(./main)

    exec_time=$(echo $output | cut -d',' -f2)
    memory_time=$(echo $output | cut -d',' -f3)

    echo "PAE | Code: $1 | Execution: $i | Exec time: $exec_time seconds | Memory time: $memory_time seconds"

    total_exec_time=$(echo "scale=6; $total_exec_time + $exec_time" | bc | awk '{printf "%.6f\n", $0}')
    total_memory_time=$(echo "scale=6; $total_memory_time + $memory_time" | bc | awk '{printf "%.6f\n", $0}')

done

avg_exec_time=$(echo "scale=6; $total_exec_time / $max_iter" | bc | awk '{printf "%.6f\n", $0}')
avg_memory_time=$(echo "scale=6; $total_memory_time / $max_iter" | bc | awk '{printf "%.6f\n", $0}')

echo "PAE | Code: $1 | Total: $max_iter | Avg Time: $avg_exec_time seconds | Avg Memory: $avg_memory_time"

rm main
