#!/bin/bash

#python -m examples.C_Arm_Denoising.train_denoising \
#       --dataset_name 'dicom_for_comparing' \
#       --dataset_type 'paired_dicom' \
#       --model 'otcfm' \
#       --condition True \
#       --batch_size 8 \
#       --num_workers 8 \
#       --save_step 10000

python -m examples.C_Arm_Denoising.train_denoising \
       --training_name 'otcfm_publicdata' \
       --dataset_name 'Denoising-Low-dose-images--main/dataset' \
       --dataset_type 'paired_jpeg' \
       --model 'otcfm' \
       --batch_size 32 \
       --num_workers 8 \
       --save_step 10000

# sed -i 's/\r$//' run_train.sh