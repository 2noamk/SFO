#!/bin/bash
mkdir -p /log_files
for i in 2 3 4 5 6 7 8 9 10; do 
  logfile="/log_files/sw2_${i}.txt"
  echo "Submitting job with kernel_layers=${i}"

  bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32 \
    -o "$logfile" \
    python -u /sallow_water/svd2d_sw.py\
      --kernel_layers "$i" 
done
echo "All jobs submitted."
