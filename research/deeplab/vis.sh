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
# This script is used to run digs dataset
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"
POLAR_FOLDER="polar"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
OM_FOLDER="/om/user/amineh/pretrained"
#INIT_FOLDER="${WORK_DIR}/saved/deeplabv3_pascal_trainval/model.ckpt"
INIT_FOLDER="${OM_FOLDER}/deeplabv3_pascal_trainval/model.ckpt"
TRAIN_LOGDIR="${OM_FOLDER}/log/polar"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${POLAR_FOLDER}/vis"

POLAR_DATASET="${WORK_DIR}/${DATASET_DIR}/${POLAR_FOLDER}/tfrecord"


# From tensorflow/models/research/
python "${WORK_DIR}"/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size="42, 42" \
    --fine_tune_batch_norm = True \
    --dataset="polar" \
    --checkpoint_dir="${TRAIN_LOGDIR}" \
    --vis_logdir="${VIS_LOGDIR}" \
    --dataset_dir="${POLAR_DATASET}" \
    

