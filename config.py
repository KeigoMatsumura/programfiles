import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_IMAGE_DIR = "../data/coco/train"
VAL_IMAGE_DIR = "../data/coco/val"
LEARNING_RATE = 2e-4 #added
D_LEARNING_RATE = 2e-4 # original 2e-4
G_LEARNING_RATE = 1e-4
BATCH_SIZE = 1
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
NUM_EPOCHS = 60
CE_LAMBDA = [100,30]
L2_LAMBDA = [1, 0.03]
LOAD_MODEL = True 
SAVE_MODEL = True
CHECKPOINT_DISC = "./result5/disc_1_CE_100_epoch_55.pth.tar" #"./model/discriminator.pth.tar" #"./result5/disc_1_CE_100_epoch_55.pth.tar"
CHECKPOINT_GEN = "./result5/gen_L1_1_CE_100_epoch_55.pth.tar" #"./model/generator.pth.tar" #"./result5/gen_L1_1_CE_100_epoch_55.pth.tar"
#"./result2/gen_L1_1_CE_100_epoch_30.pth.tar"
#"generator.pth.tar"
