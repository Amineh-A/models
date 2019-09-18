
BATCH_SIZE=32

FOR_COUNTER=0
DONE_EXPS=0

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
DIGS_FOLDER="digs"
OM_DIR="/om/user/amineh"
#OM_DIR="/Users/amineh.ahm/Desktop/InsideOutside/om"
INIT_DIR="${OM_DIR}/pretrained/deeplabv3_pascal_trainval"

for LEARNING_RATE in 0.01
do
    for ALPHA in 0.2
    do
        for OUTPUT_STRIDE in 8
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


	    INDEX="${BATCH_SIZE}_${LEARNING_RATE}_${ALPHA}_${OUTPUT_STRIDE}"

            TRAIN_LOGDIR="${OM_DIR}/exp/polar_constant_lr_5/${INDEX}/train"
            EXP_DIR="${OM_DIR}/exp/digs/${INDEX}"
            EVAL_LOGDIR="${EXP_DIR}/eval/"
            VIS_LOGDIR="${EXP_DIR}/vis"
            mkdir -p "${EXP_DIR}"
            mkdir -p "${EVAL_LOGDIR}"
            mkdir -p "${VIS_LOGDIR}"
            DIGS_DATASET="${WORK_DIR}/${DATASET_FOLDER}/${DIGS_FOLDER}/tfrecord"


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
              --dataset="digs" \
              --eval_batch_size="${BATCH_SIZE}" \
              --dataset_dir="${DIGS_DATASET}" \
              --max_number_of_evaluations=1 \
              --gpu="${GPU}" \

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
              --dataset="digs" \
              --dataset_dir="${DIGS_DATASET}" \
              --max_number_of_iterations=1 \
              --gpu="${GPU}" \

	    fi
        done
    done
done
