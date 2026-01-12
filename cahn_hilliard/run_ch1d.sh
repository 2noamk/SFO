#!/bin/bash
mkdir -p /log_files

for i in 3 5 7 9; do
  for l in 2 4 6; do
    for k_exp in 5 7; do
      for j_exp in 5 7; do
        k=$((1 << k_exp))
        j=$((1 << j_exp))
        logfile="/log_files/ch1_${i}_${l}_${k}_${j}.txt"
        echo "Submitting job with num_eigenfunc=${i}, N_H=${l}, H=${k}, lifting_dim=${j}"

        bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32 \
          -o "$logfile" \
          python -u /cahn_hilliard/svd1d_ch.py \
            --num_eigenfunc "$i" \
            --N_H "$l" \
            --H "$k" \
            --lifting_dim "$j"
      done
    done
  done
done
echo "All jobs submitted."
