#!/bin/bash
# Example of use: ./scripts/script_02_job.sh 04_distance.c
# Parse results: ./parser.sh ./outs/P3_2.o ./data/data_1.csv

rm -r ./outs
mkdir ./outs

module load gcc

gcc -O2 -fopenmp -lm -o main $1

cores=(1 2 4 8 16 32 48 64)

for i in ${cores[@]}
do
	sbatch -J P3_$i -o ./outs/P3_$i.o -e ./outs/P3_$i.e -N 1 -n $i --mem=128GB --time=02:00:00 ./scripts/script_02_task.sh main
done