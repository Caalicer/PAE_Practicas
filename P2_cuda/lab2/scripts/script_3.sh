#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH --gres gpu:a100:1
#SBATCH -t 00:02:00
#SBATCH -J P2_1
#SBATCH -o ./outs/P2_1.o
#SBATCH -e ./outs/P2_1.e

# Valid for exercises [daxpy]

nvcc -O2 -o main $1

block_sizes=(1 2 4 8 16 32 64 128 256 512 1024)

max_iter=10

for block in ${block_sizes[@]}; do

	for i in $(seq 1 $max_iter); do

		output=$(./main $block)
		echo $output

	done

done

rm main
