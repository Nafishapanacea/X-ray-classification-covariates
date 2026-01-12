import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

CSV_PATH = "data/metadata.csv"

NUM_VIEW_TYPES = 3 # PA, AP, Lateral
NUM_SEX_TYPES = 2 # M, F