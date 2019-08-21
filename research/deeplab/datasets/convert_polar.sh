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
# Script to preprocess the Digs dataset.
# By changing convert_cityscapes
#
#
# The folder structure is assumed to be:
#  + datasets
#    - build_digs_data.py
#    - convert_digs.sh
#    + digs (?)
#      + cityscapesscripts (downloaded scripts)
#      + gtFine
#      + leftImg8bit
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="."

# Root path for digs dataset.
POLAR_ROOT="${WORK_DIR}/polar"

# Create training labels.
# ?
# python "${DIGS_ROOT}/cityscapesscripts/preparation/createTrainIdLabelImgs.py"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${POLAR_ROOT}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

BUILD_SCRIPT="${CURRENT_DIR}/build_polar_data.py"

echo "Converting polar dataset..."
python "${BUILD_SCRIPT}" \
  --output_dir="${OUTPUT_DIR}" \
  --image_format="png" \
