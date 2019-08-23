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
OM_FOLDER="/om/user/amineh/pretrained/log/polar"
#INIT_FOLDER="${WORK_DIR}/saved/deeplabv3_pascal_trainval/model.ckpt"
INIT_FOLDER="${OM_FOLDER}/deeplabv3_pascal_trainval/model.ckpt"
TRAIN_LOGDIR="${OM_FOLDER}/train"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${POLAR_FOLDER}/vis"

POLAR_DATASET="${WORK_DIR}/${DATASET_DIR}/${POLAR_FOLDER}/tfrecord"

# Export the trained checkpoint.
NUM_ITERATIONS=100
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${OM_FOLDER}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --dataset="polar" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=3 \
  --crop_size=42 \
  --crop_size=42 \
  --inference_scales=1.0
