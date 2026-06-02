IMG_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 4
EPOCHS = 30
LEARNING_RATE = 0.0001

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

# Model save paths (consistent with training script)
MODEL_SAVE_PATH_CBAM = 'CBAM_model_best.pth'
MODEL_SAVE_PATH_SE   = 'SE_model_best.pth'
MODEL_SAVE_PATH_CNN  = 'CNN_model_best.pth'   # <-- add this

TRAIN_ACC_FILE = "train_accuracy.txt"
TEST_ACC_FILE = "test_accuracy.txt"
