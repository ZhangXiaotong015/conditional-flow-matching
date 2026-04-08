#!/bin/bash

python -m examples.C_Arm_Denoising.train_denoising \
       --dataset_name 'dicom_for_comparing' \
       --model 'otcfm' \
       --batch_size 8 \
       --num_workers 4 \
       --save_step 10000

# sed -i 's/\r$//' run_train.sh