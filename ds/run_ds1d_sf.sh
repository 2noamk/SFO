#!/bin/bash
mkdir -p /log_files
for m in 4 8 16 32 64 128; do
  for w in 32 64 128 256; do
    logfile="/log_files/ds1_${m}_${w}_sf.txt"
    echo "Submitting job with kernel_layers=${m}_${w}"

    bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32 \
      -o "$logfile" \
      python -u /ds/sno_ds.py \
        --modes "$m" \
        --width "$w"
  done
done
echo "All jobs submitted."
