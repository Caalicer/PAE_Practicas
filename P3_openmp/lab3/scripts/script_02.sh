#!/bin/bash

#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem 128G
#SBATCH -t 02:00:00
#SBATCH -J P3_2
#SBATCH -o ./outs/P3_2.o
#SBATCH -e ./outs/P3_2.e

module load gcc

gcc -O2 -fopenmp -lm -o main $1

cores=(1 2 4 8 16 32 48 64)
max_ite=10

for core in ${cores[@]}; do

	i=1

	while [ $i -le $max_ite ]; do

		./main $core
		i=$((i+1))

	done

done

rm main