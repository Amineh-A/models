#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run train, eval, vis on polar_spiral dataset
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

BATCH_SIZE=32
NUM_ITERATIONS=30000

FOR_COUNTER=0
DONE_EXPS=4

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working directories.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_FOLDER="datasets"
POLAR_FOLDER="polar"
OM_DIR="/om/user/amineh"
#OM_DIR="/Users/amineh.ahm/Desktop/InsideOutside/om"
INIT_DIR="${OM_DIR}/pretrained/deeplabv3_pascal_trainval"

for LEARNING_RATE in 0.1
do
    for ALPHA in 0.1 0.2 0.4
    do
        for OUTPUT_STRIDE in 8 16
        do
            if [ "$OUTPUT_STRIDE" -eq 8 ]
            then
                ATROUS_0=12
                ATROUS_1=24
                ATROUS_2=36
            else
                ATROUS_0=6
                ATROUS_1=12
                ATROUS_2=18
            fi
		
	    FOR_COUNT=$((FOR_COUNT+1))
            if [ "$FOR_COUNT" -gt "$DONE_EXPS" ]
	    then
		

	    INDEX="${BATCH_SIZE}_0.1_${ALPHA}_${OUTPUT_STRIDE}"

            EXP_DIR="${OM_DIR}/exp/polar_constant_lr_6/${INDEX}"
            TRAIN_LOGDIR="${EXP_DIR}/train"
            EVAL_LOGDIR="${EXP_DIR}/eval"
            VIS_LOGDIR="${EXP_DIR}/vis"
            EXPORT_DIR="${EXP_DIR}/export"
            mkdir -p "${EXP_DIR}"
            mkdir -p "${TRAIN_LOGDIR}"
            mkdir -p "${EVAL_LOGDIR}"
            mkdir -p "${VIS_LOGDIR}"
            mkdir -p "${EXPORT_DIR}"
            POLAR_DATASET="${WORK_DIR}/${DATASET_FOLDER}/${POLAR_FOLDER}/tfrecord"


            # Train
            python "${WORK_DIR}"/train.py \
              --logtostderr \
              --train_split="train" \
              --model_variant="xception_65" \
              --atrous_rates="${ATROUS_0}" \
              --atrous_rates="${ATROUS_1}" \
              --atrous_rates="${ATROUS_2}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --decoder_output_stride=4 \
              --train_crop_size="42, 42" \
              --train_batch_size="${BATCH_SIZE}" \
              --training_number_of_steps="${NUM_ITERATIONS}" \
              --dataset="polar" \
              --tf_initial_checkpoint="${INIT_DIR}/model.ckpt" \
              --train_logdir="${TRAIN_LOGDIR}" \
              --dataset_dir="${POLAR_DATASET}" \
              --initialize_last_layer=False \
              --last_layers_contain_logits_only=True \
              --loss_weight_alpha="${ALPHA}" \
              --base_learning_rate="${LEARNING_RATE}" \


            # Run evaluation. This performs eval over the full val split (1449 images) and
            # will take a while.
            # Using the provided checkpoint, one should expect mIOU=82.20%.
            python "${WORK_DIR}"/eval.py \
              --logtostderr \
              --eval_split="trainval" \
              --model_variant="xception_65" \
              --atrous_rates="${ATROUS_0}" \
              --atrous_rates="${ATROUS_1}" \
              --atrous_rates="${ATROUS_2}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --decoder_output_stride=4 \
              --eval_crop_size="42, 42" \
              --checkpoint_dir="${TRAIN_LOGDIR}" \
              --eval_logdir="${EVAL_LOGDIR}" \
              --dataset="polar" \
              --eval_batch_size="${BATCH_SIZE}" \
              --dataset_dir="${POLAR_DATASET}" \
              --max_number_of_evaluations=1

            # Visualize the results.
            python "${WORK_DIR}"/vis.py \
              --logtostderr \
              --vis_split="vis" \
              --model_variant="xception_65" \
              --atrous_rates="${ATROUS_0}" \
              --atrous_rates="${ATROUS_1}" \
              --atrous_rates="${ATROUS_2}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --decoder_output_stride=4 \
              --vis_crop_size="42, 42" \
              --checkpoint_dir="${TRAIN_LOGDIR}" \
              --vis_logdir="${VIS_LOGDIR}" \
              --dataset="polar" \
              --dataset_dir="${POLAR_DATASET}" \
              --max_number_of_iterations=1

            # Export the trained checkpoint.
            CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
            EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

            python "${WORK_DIR}"/export_model.py \
              --logtostderr \
              --checkpoint_path="${CKPT_PATH}" \
              --export_path="${EXPORT_PATH}" \
              --model_variant="xception_65" \
              --atrous_rates="${ATROUS_0}" \
              --atrous_rates="${ATROUS_1}" \
              --atrous_rates="${ATROUS_2}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --decoder_output_stride=4 \
              --num_classes=3 \
              --crop_size=42 \
              --crop_size=42 \
              --dataset="polar" \
              --inference_scales=1.0

            # Run inference with the exported checkpoint.
            # Please refer to the provided deeplab_demo.ipynb for an example.

	    fi
        done
    done
done
