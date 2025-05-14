#!/bin/bash

#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem 128G
#SBATCH -t 02:00:00
#SBATCH -J P3_conv
#SBATCH -o ./outs/P3_conv.o
#SBATCH -e ./outs/P3_conv.e

module load gcc

gcc -O2 -fopenmp -lm -o main $1

cores=(1 2 4 8 16 32 48 64)
max_ite=10

for core in ${cores[@]}; do

    for image in $(ls ./images/*.pgm); do

		for i in $(seq 1 $max_iter); do

			output=$(./main $image $core)
			echo -e $output

		done
	done
done

./scripts/parser.sh ./outs/P3_conv.o ./data/data_conv

rm main