import os

from easydict import EasyDict as edict

cfg_data = edict()

cfg_data.TRAIN_SIZE = (768, 1024)
cfg_data.TRAINING_MAX_LONG = 2560
cfg_data.TRAINING_MAX_SHORT = 1440
cfg_data.TEST_MAX_LONG = 1920
cfg_data.TEST_MAX_SHORT = 1080
cfg_data.DATA_PATH = os.environ.get("MDC_DATA_ROOT", "./data/MovingDroneCrowd")
cfg_data.FLOW_ROOT = os.environ.get("MDC_FLOW_ROOT", os.path.join(cfg_data.DATA_PATH, "raft"))
cfg_data.TRAIN_LST = "train.txt"
cfg_data.VAL_LST = "val.txt"
cfg_data.TEST_LST = "test.txt"

cfg_data.MEAN_STD = (
    [117 / 255.0, 110 / 255.0, 105 / 255.0],
    [67.10 / 255.0, 65.45 / 255.0, 66.23 / 255.0],
)

cfg_data.DEN_FACTOR = 200.0
cfg_data.RESUME_MODEL = ""
cfg_data.TRAIN_BATCH_SIZE = 1
cfg_data.TRAIN_FRAME_INTERVALS = (3, 8)
cfg_data.VAL_FRAME_INTERVALS = 4
cfg_data.VAL_BATCH_SIZE = 1
