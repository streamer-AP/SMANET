import os
import time

from easydict import EasyDict as edict

cfg = edict()

# Reproducibility and dataset selection.
cfg.SEED = int(os.environ.get("VIC_SEED", "3035"))
cfg.DATASET = os.environ.get("VIC_DATASET", "MovingDroneCrowd")
cfg.NAME = os.environ.get("VIC_EXP_NAME", "default")  # Optional experiment suffix.
cfg.encoder = "VGG16_FPN"
cfg.INPUT_STRIDE = int(os.environ.get("VIC_INPUT_STRIDE", "32"))  # Model input padding stride.

# Checkpoint and pretrained-model options.
cfg.RESUME = os.environ.get("VIC_RESUME", "0") == "1"
cfg.RESUME_PATH = os.environ.get("VIC_RESUME_PATH", "")
cfg.PRE_TRAIN_COUNTER = os.environ.get("VIC_PRETRAIN_COUNTER", "")

# Device selection. DDP sets the per-process device in train.py.
cfg.GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID

# SMANet architecture.
cfg.cross_attn_embed_dim = 256
cfg.cross_attn_num_heads = 4
cfg.mlp_ratio = 4
cfg.cross_attn_depth = 2
cfg.FEATURE_DIM = 256

# Optimizer settings. New probability/flow modules use a larger learning rate.
cfg.NEW_MODULE_LR = float(os.environ.get("VIC_NEW_MODULE_LR", "2e-5"))
cfg.BASE_MODULE_LR = float(os.environ.get("VIC_BASE_MODULE_LR", "5e-6"))
cfg.LR_POWER = float(os.environ.get("VIC_LR_POWER", "0.9"))  # Polynomial decay power.
cfg.WEIGHT_DECAY = float(os.environ.get("VIC_WEIGHT_DECAY", "1e-6"))

# Training schedule and objective weights.
# The probability branch is warmed up with detached image features and BCE first.
cfg.FEATURE_FREEZE_EPOCHS = int(os.environ.get("VIC_FEATURE_FREEZE_EPOCHS", "20"))
cfg.PROBABILITY_BCE_EPOCHS = int(os.environ.get("VIC_PROBABILITY_BCE_EPOCHS", "5"))
cfg.GLOBAL_LOSS_WEIGHT = float(os.environ.get("VIC_GLOBAL_LOSS_WEIGHT", "1.0"))
cfg.SHARE_LOSS_WEIGHT = float(os.environ.get("VIC_SHARE_LOSS_WEIGHT", "10.0"))
cfg.IN_OUT_LOSS_WEIGHT = float(os.environ.get("VIC_IN_OUT_LOSS_WEIGHT", "1.0"))
cfg.PROBABILITY_LOSS_WEIGHT = float(os.environ.get("VIC_PROBABILITY_LOSS_WEIGHT", "0.1"))
cfg.PROBABILITY_BCE_WEIGHT = float(os.environ.get("VIC_PROBABILITY_BCE_WEIGHT", "0.3"))
cfg.PROBABILITY_FOCAL_WEIGHT = float(os.environ.get("VIC_PROBABILITY_FOCAL_WEIGHT", "0.7"))
cfg.PROBABILITY_POS_WEIGHT = float(os.environ.get("VIC_PROBABILITY_POS_WEIGHT", "10.0"))

# Focal-Tversky parameters are separate from the BCE/Focal loss mixture weights.
cfg.FOCAL_TVERSKY_ALPHA = float(os.environ.get("VIC_FOCAL_TVERSKY_ALPHA", "0.3"))
cfg.FOCAL_TVERSKY_BETA = float(os.environ.get("VIC_FOCAL_TVERSKY_BETA", "0.7"))
cfg.FOCAL_TVERSKY_GAMMA = float(os.environ.get("VIC_FOCAL_TVERSKY_GAMMA", "0.5"))
cfg.FOCAL_TVERSKY_SMOOTH = float(os.environ.get("VIC_FOCAL_TVERSKY_SMOOTH", "1e-5"))

# Target-map generation settings.
cfg.PROBABILITY_SIGMA = float(os.environ.get("VIC_PROBABILITY_SIGMA", "4.0"))
cfg.PROBABILITY_KERNEL_SIZE = int(os.environ.get("VIC_PROBABILITY_KERNEL_SIZE", "31"))
cfg.DENSITY_SIGMA = float(os.environ.get("VIC_DENSITY_SIGMA", "4.0"))
cfg.DENSITY_KERNEL_SIZE = int(os.environ.get("VIC_DENSITY_KERNEL_SIZE", "15"))

# Pair sampling, visualization, and distributed-training behavior.
cfg.MIN_SHARED_PEOPLE = int(os.environ.get("VIC_MIN_SHARED_PEOPLE", "3"))
cfg.TRAIN_SCALE_MIN = float(os.environ.get("VIC_TRAIN_SCALE_MIN", "0.8"))
cfg.TRAIN_SCALE_MAX = float(os.environ.get("VIC_TRAIN_SCALE_MAX", "1.2"))
cfg.TRAIN_VIS_INTERVAL = int(os.environ.get("VIC_TRAIN_VIS_INTERVAL", "100"))
cfg.SAVE_VAL_VISUAL = os.environ.get("VIC_SAVE_VAL_VISUAL", "0") == "1"
cfg.DDP_FIND_UNUSED_PARAMETERS = os.environ.get("VIC_DDP_FIND_UNUSED_PARAMETERS", "1") == "1"

# Runtime and validation settings.
cfg.MAX_EPOCH = int(os.environ.get("VIC_MAX_EPOCH", "60"))
cfg.VAL_INTERVAL = int(os.environ.get("VIC_VAL_INTERVAL", "1"))
cfg.START_VAL = int(os.environ.get("VIC_START_VAL", "0"))
cfg.PRINT_FREQ = int(os.environ.get("VIC_PRINT_FREQ", "20"))
cfg.TRAIN_NUM_WORKERS = int(os.environ.get("VIC_TRAIN_NUM_WORKERS", "32"))
cfg.VAL_NUM_WORKERS = int(os.environ.get("VIC_VAL_NUM_WORKERS", "8"))
cfg.SMOKE_MAX_ITERS = int(os.environ.get("VIC_SMOKE_MAX_ITERS", "0"))

# Output paths are created relative to the code directory when training starts.
now = time.strftime("%m-%d_%H-%M", time.localtime())
cfg.EXP_NAME = f"{now}_{cfg.DATASET}_{cfg.NEW_MODULE_LR}_{cfg.NAME}"
cfg.VAL_VIS_PATH = os.path.join("./exp", f"{cfg.DATASET}_val")
cfg.EXP_PATH = os.path.join("./exp", cfg.DATASET)
os.makedirs(cfg.EXP_PATH, exist_ok=True)
