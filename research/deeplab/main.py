from deeplab import train, eval, vis, export_model
from sacred import Experiment
from labwatch.assistant import LabAssistant
from labwatch.optimizers import RandomSearch
import os

ex = Experiment()
a = LabAssistant(experiment=ex,
                 database_name="labwatch",
                 optimizer=RandomSearch)


@ex.config
def config():
    num_iterations = 100
    train_batch_size = 4
    model_variants = "xception_65"
    atrous_rates_0 = 6
    atrous_rates_1 = 12
    atrous_rates_2 = 18
    output_stride = 16
    decoder_output_stride = 4

    crop_size = "42, 42"
    fine_tune_batch_norm = True

    exp_id = 0
    OM_DIR = "/om/user/amineh/exp/exp_id"
    INIT_DIR = "${OM_DIR}/deeplabv3_pascal_trainval/model.ckpt"
    TRAIN_LOGDIR = "${OM_DIR}/train"
    VIS_LOGDIR = "${OM_DIR}/vis"
    CKPT_PATH = "${TRAIN_LOGDIR}/model.ckpt-${num_iterations}"
    EXPORT_PATH = "${OM_DIR}/frozen_inference_graph.pb"

    WORK_DIR = "/om/user/amineh/models/research/deeplab"
    DATASET_DIR = "${WORK_DIR}/datasets"
    POLAR_DIR = "${DATASET_DIR}/polar"
    POLAR_DATASET = "${POLAR_DIR}/tfrecord"


@a.searchspace
def search_space():
    """ search space to find batchsize randomly """
    train_batch_size = UniformInt(lower=12, upper=32, log_scale=True)
    output_stride = Categorical({8, 16})


@ex.capture
def setup_output_stride(output_stride):
    """ atrous_rates based on output stride """
    if output_stride is 16:
        return 6, 12, 18, 16
    else:
        return 12, 24, 36, 8


@ex.automain
def run():










