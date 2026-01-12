#!/bin/bash
mkdir -p /log_files
for e in 2 3 4 5; do
  for d in 32 64 128 256 512; do
    for k in 2 3 4 5 6 7; do
      for lr in 0.1 0.01; do
        logfile="/log_files/ds1_hno_${e}_${d}_${k}_${lr}.txt"
        echo "Submitting job ds1_hno_${e}_${d}_${k}_${lr}"

        bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32 \
          -o "$logfile" \
          python -u /ds/hno_ds.py \
            --num_eigenfunc "$e"\
            --lifting_dim "$d"\
            --kernel_layers "$k"\
            --learning_rate "$lr"
      done
    done
  done     
done
echo "All jobs submitted."
