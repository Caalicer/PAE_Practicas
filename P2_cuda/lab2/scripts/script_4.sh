#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH --gres gpu:a100:1
#SBATCH -t 00:02:00
#SBATCH -J P2_1
#SBATCH -o ./outs/P2_1.o
#SBATCH -e ./outs/P2_1.e

# Valid for exercises [histogram_1, histogram_2, convolution_1]

nvcc -O2 -o main $1

block_sizes=(
	"128 8"
	"64 16"
	"32 32"
	"16 64"
	"8 128"
)

max_iter=10

for block in "${block_sizes[@]}"; do

	for image in $(ls ./images/*.pgm) do

		for i in $(seq 1 $max_iter) do

			output=$(./main $image $block)
			echo $output

		done

	done

done

rm main
