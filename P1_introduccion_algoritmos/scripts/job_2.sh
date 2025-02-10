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

total_time=0
max_iter=10

for image in $(ls ./images/*.pgm)
do

	sum_time=0

	for i in $(seq 1 $max_iter)
	do

		output=$(./main $image)
		time=$(echo "$output" | grep "PAE | Time:" | awk '{print $4}')
		echo "PAE | Code: $1 | Image: $image | Execution $i: $time seconds"
		sum_time=$(echo "$sum_time + $time" | bc)
		total_time=$(echo "$total_time + $time" | bc)

	done

	avg_time=$(echo "scale=6; $sum_time / $max_iter" | bc)
	echo "PAE | Code: $1 | Image: $image | Total: $sum_time | Average: $avg_time seconds"

done

avg_time=$(echo "scale=6; $total_time / ($max_iter * $(ls ./images/*.pgm | wc -l))" | bc)
echo "PAE | Code: $1 | Total: $total_time | Average: $avg_time seconds"

rm main
