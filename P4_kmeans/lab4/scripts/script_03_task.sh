#!/bin/bash

CU_FILE=$1

for i in {0..9} do
    sbatch \
        --job-name=P4_3_$i \
        --output=./outs/P4_3_$i.o \
        --error=./outs/P4_3_$i.e \
        --nodes=1 \
        --cpus-per-task=32 \
        --mem=64G \
        --gres=gpu:a100:1 \
        --time=00:45:00 \
        job_runner.sh "$CU_FILE" "$i"
done
