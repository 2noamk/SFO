#!/bin/bash
mkdir -p /log_files

for i in 2 3 4 5 6 7 8 9 10; do
    logfile="/log_files/dr1_${i}.txt"
    echo "Submitting job with kernel_layers=${i}"

    bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32 \
      -o "$logfile" \
      python -u /dr/svd1d_dr.py \
        --kernel_layers "$i" 
done
echo "All jobs submitted."
