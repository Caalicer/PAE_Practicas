#!/bin/bash

#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem 128G
#SBATCH -t 24:00:00
#SBATCH -J P4_2
#SBATCH -o ./outs/P4_2.o
#SBATCH -e ./outs/P4_2.e

module load gcc

gcc -Wall -O2 -fopenmp -lm -o main_2 $1

cores=(1 2 4 8 16 32 48 64)
max_iter=10

for core in ${cores[@]}; do

	for image in $(ls ./images/*.raw); do

		for i in $(seq 1 $max_iter); do

			./main_2 $image $core

		done

	done

done

rm main_2
