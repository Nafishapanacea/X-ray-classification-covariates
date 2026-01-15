import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4

img_dir = '/home/common/data_v3'
train_csv = '/home/jupyter-nafisha/X-ray-covariates/CSVs/train.csv'
val_csv= '/home/jupyter-nafisha/X-ray-covariates/CSVs/val.csv'

NUM_VIEW_TYPES = 3 # PA, AP, Lateral
NUM_SEX_TYPES = 2 # M, F