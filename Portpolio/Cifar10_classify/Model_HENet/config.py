import os
import glob

### Path Info ###
DATA_PATH = 'C:\\python\\source\\Portpolio\\Cifar10_classify\\data\\cifar10_dataset'
WORKING_DIR_PATH = 'C:\\python\\source\\Portpolio\\Cifar10_classify\\Model_HENet'
CKPT_DIR_PATH = os.path.join(WORKING_DIR_PATH, 'log')
CKPT_FILE = os.path.join(CKPT_DIR_PATH, 'save.ckpt')

### Export or Restore Checkpoint ###
RESTORE = False
EXPORT = False

### Hyper Parameter ###

# common
EPOCHS = 100                     # total train epochs
SAVE_EPOCHS = 1                  # saving terms
BATCH_SIZE = 100                 # batch_size
LABEL_CNT = 10                    # label's class count (dog or cat -> 2)
OPTIMIZER = 'adam'               # Adam / RMSProp (Modify vggnet.py/ select_optimizer to add new optimizer)
ACTIVATION_FN = 'elu'            # Swish / elu (Modify vggnet.py/ select_activation_fn to add new activation_fn)
LEARNING_RATE = 3e-3             # initial learning rate
DECAY_STEPS = 1000               # learning rate decay step
DECAY_RATE = 0.9                 # learning rate decay rate
STAIRCASE = False                 # learning rate decay staircase
BATCHNORM_DECAY_RATE = 0.9       # batch norm decay rate (default : 0.999 / if good train & bad validation : 0.9 recommended)
L2_REG_RATE = 1e-3               # l2 regularization rate
DROPOUT_KEEP_PROB = 0.5

# HENet