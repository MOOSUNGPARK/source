import os
import glob

### Path Info ###
TRAIN_DATA_PATH = 'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\train'
TEST_DATA_PATH = 'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\test'
VALIDATION_DATA_PATH = 'C:\\python\\source\\Portpolio\\Dog_Cat\\data\\validation'

TRAIN_FILE_LIST = glob.glob(os.path.join(TRAIN_DATA_PATH, '*.csv'), recursive=False)
TEST_FILE_LIST = glob.glob(os.path.join(TEST_DATA_PATH, '*.csv'), recursive=False)
VALIDATION_FILE_LIST = glob.glob(os.path.join(VALIDATION_DATA_PATH, '*.csv'), recursive=False)

WORKING_DIR_PATH = 'C:\\python\\source\\Portpolio\\Dog_Cat\\Model_vggnet\\renewal'
CKPT_DIR_PATH = os.path.join(WORKING_DIR_PATH, 'log')
CKPT_FILE = os.path.join(CKPT_DIR_PATH, 'save.ckpt')

### Restore Checkpoint ###
RESTORE = False

### Hyper Parameter ###

# common
EPOCHS = 50                      # total train epochs
SAVE_EPOCHS = int(EPOCHS * 0.02)  # saving terms (default 0.2) 50 total epochs -> save 10, 20, ..,50 epoch's info
BATCH_SIZE = 100                 # batch_size
LABEL_CNT = 2                    # label's class count (dog or cat -> 2)
OPTIMIZER = 'adam'               # Adam / RMSProp (Modify vggnet.py/ select_optimizer() to add new optimizer)
ACTIVATION_FN = 'swish'          # Swish / elu / relu (Modify vggnet.py/ select_activation_fn() to add new activation_fn)
LEARNING_RATE = 1e-3             # initial learning rate
DECAY_STEPS = 5000               # learning rate decay step
DECAY_RATE = 0.1                 # learning rate decay rate
STAIR_CASE = True                 # learning rate decay staircase
BATCHNORM_DECAY_RATE = 0.99      # batch norm decay rate (default : 0.999 / if good train & bad validation : 0.9 recommended)
L2_REG_RATE = 1e-4               # l2 regularization rate
DROPOUT_KEEP_PROB = 0.5          # dropout rate

# model
