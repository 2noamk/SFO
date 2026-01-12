#!/bin/bash
mkdir -p /log_files
for i in 3 5 6 7 8 9 10; do
  for j in 
    logfile="/log_files/ds1_${i}.txt"
    echo "Submitting job with kernel_layers=${i}"

    bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32 \
      -o "$logfile" \
      python -u /ds/svd1d_no_integral_ds.py \
        --num_eigenfunc "$i"\
        --N_H "$i"\
        --H "$i"\
        --lifting_dim "$i"
        
done
echo "All jobs submitted."
