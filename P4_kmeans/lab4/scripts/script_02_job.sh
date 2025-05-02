#!/bin/bash

module load gcc

gcc -Wall -O2 -fopenmp -lm -o main_2 $1

#cores=(1 2 4 8 16 32 48 64)
#done 1 2 4 8 16 32 48 64

cores=(64)
schedulers=(0 1 2 3)
chunk_sizes=(1 2 4 8 16 32 64 128 256 512)

dest="./outs/"

for image in $(ls ./images/*.raw); do

	filename=$(basename "$image")

	for core in ${cores[@]}; do

		for scheduler in ${schedulers[@]}; do

			for chunk_size in ${chunk_sizes[@]}; do

				name="P4_2_${filename}_${core}_${scheduler}_${chunk_size}"

				sbatch -J "${name}" -o "${dest}${name}.o" -e "${dest}${name}.e" -c $core ./scripts/script_02_task.sh main_2 $image $core $scheduler $chunk_size

			done

		done

	done

done
