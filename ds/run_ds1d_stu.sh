#!/bin/bash
mkdir -p /log_files
for d in 1 5 10 15 24; do
  for k in 1 2 3 4 5 6; do
    for e in 1 2 4 8 16 24; do
      logfile="/log_files/ds1_stu_${e}_${d}_${k}.txt"
      echo "Submitting job ds1_hno_${e}_${d}_${k}"

      bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32 \
        -o "$logfile" \
        python -u /ds/stu_ds.py \
          --num_eigenfunc "$e"\
          --lifting_dim "$d"\
          --kernel_layers "$k"
    done
  done     
done
echo "All jobs submitted."
