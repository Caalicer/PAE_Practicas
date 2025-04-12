#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 32G
#SBATCH -t 24:00:00
#SBATCH -J P4_1
#SBATCH -o ./outs/P4_1.o
#SBATCH -e ./outs/P4_1.e

module load gcc

gcc -Wall -O2 -lm -o main_1 $1

max_iter=10

for image in $(ls ./images/*.raw); do

	for i in $(seq 1 $max_iter); do

		./main_1 $image

	done

done

rm main_1
