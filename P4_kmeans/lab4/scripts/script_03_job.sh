#!/bin/bash

CUFILE=$1
ITER=$2

# Compilar el programa CUDA
nvcc -O2 -o "main_cuda_${ITER}" "$CUFILE"

block_sizes=(1 2 4 8 16 32 64 128 256 512 1024)

for block in "${block_sizes[@]}"; do

    for image in ./images/*.raw; do

        output=$(./main_cuda_${ITER} "$image" "$block")
        echo -e "$output"

    done

done

# Parseo del output específico de esta iteración
./scripts/parser.sh "./outs/P4_3_${ITER}.o" "./data/data_${ITER}.csv"

# Limpieza
rm "main_cuda_${ITER}"
