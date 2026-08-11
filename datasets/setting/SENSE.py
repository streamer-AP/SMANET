import os

from easydict import EasyDict as edict

cfg_data = edict()

cfg_data.TRAIN_SIZE = (768, 1024)
cfg_data.TRAINING_MAX_LONG = 2560
cfg_data.TRAINING_MAX_SHORT = 1440
cfg_data.TEST_MAX_LONG = 1920
cfg_data.TEST_MAX_SHORT = 1080
cfg_data.DATA_PATH = os.environ.get("SENSE_DATA_ROOT", "./data/SENSE")
cfg_data.FLOW_ROOT = os.environ.get("SENSE_FLOW_ROOT", "./data/SENSE/raft")
cfg_data.TRAIN_LST = "train.txt"
cfg_data.VAL_LST = "test.txt"
cfg_data.TEST_LST = "test.txt"

cfg_data.MEAN_STD = (
    [117 / 255.0, 110 / 255.0, 105 / 255.0],
    [67.10 / 255.0, 65.45 / 255.0, 66.23 / 255.0],
)

cfg_data.DEN_FACTOR = 200.0
cfg_data.RESUME_MODEL = ""
cfg_data.TRAIN_BATCH_SIZE = int(os.environ.get("VIC_TRAIN_BATCH_SIZE", "1"))
cfg_data.TRAIN_FRAME_INTERVALS = (10, 20)
cfg_data.VAL_FRAME_INTERVALS = 15
cfg_data.VAL_BATCH_SIZE = 1
