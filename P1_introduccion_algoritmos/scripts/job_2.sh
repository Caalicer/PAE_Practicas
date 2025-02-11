#!/bin/bash

# Script for running the algorithm 10 times and calculating the average time. Exercice 3 and 4.

#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 00:05:00
#SBATCH --mem 64G
#SBATCH -J P1_2
#SBATCH -o ./outs/P1.o
#SBATCH -e ./outs/P1.e

module load gcc

gcc -O2 -Wall -lm -o main $1

total_exec_time=0
total_memory_time=0

max_iter=10

for image in $(ls ./images/*.pgm)
do

    sum_exec_time=0
    sum_memory_time=0

    for i in $(seq 1 $max_iter)
    do

        output=$(./main $image | grep "PAE")

        exec_time=$(echo $output | cut -d',' -f2)
        memory_time=$(echo $output | cut -d',' -f3)

        echo "PAE | Code: $1 | Image: $image | Execution: $i | Exec time: $exec_time seconds | Memory time: $memory_time seconds"

        sum_exec_time=$(echo "$sum_exec_time + $exec_time" | bc)
        sum_memory_time=$(echo "$sum_memory_time + $memory_time" | bc)

        total_exec_time=$(echo "$total_exec_time + $exec_time" | bc)
        total_memory_time=$(echo "$total_memory_time + $memory_time" | bc)

    done

    avg_exec_time=$(echo "scale=6; $sum_exec_time / $max_iter" | bc | awk '{printf "%.6f\n", $0}')
    avg_memory_time=$(echo "scale=6; $sum_memory_time / $max_iter" | bc | awk '{printf "%.6f\n", $0}')

    echo "PAE | Code: $1 | Image: $image | Total exec time: $sum_exec_time | Total memory time: $sum_memory_time | Avg exec time: $avg_exec_time | Avg memory time: $avg_memory_time"

done

avg_exec_time=$(echo "scale=6; $total_exec_time / ($max_iter * $(ls ./images/*.pgm | wc -l))" | bc | awk '{printf "%.6f\n", $0}')
avg_memory_time=$(echo "scale=6; $total_memory_time / ($max_iter * $(ls ./images/*.pgm | wc -l))" | bc | awk '{printf "%.6f\n", $0}')

echo "PAE | Code: $1 | Total exec time: $total_exec_time | Total memory time: $total_memory_time | Avg exec time: $avg_exec_time | Avg memory time: $avg_memory_time"

rm main
