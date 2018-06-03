import os


############### DATA SET & PATH ###############
TRAIN_DATA_PATH = 'C:\\python\\source\\Portpolio\\Object_detection\\Yolo_mobilenet\\data\\train'
TEST_DATA_PATH = 'C:\\python\\source\\Portpolio\\Object_detection\\Yolo_mobilenet\\data\\test'
PASCAL_PATH = os.path.join(TRAIN_DATA_PATH, 'pascal_voc')
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')
OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')
DEVKIL_PATH = os.path.join(PASCAL_PATH, 'VOCdevkit')
VOC2007 = os.path.join(DEVKIL_PATH, 'VOC2007')
WEIGHTS_FILE = os.path.join(TRAIN_DATA_PATH, 'weights', 'save.ckpt-6000')
WEIGHTS_DIR = os.path.join(TRAIN_DATA_PATH, 'weights')

############### INFORMATION ###############
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
FLIPPED = True


############### HYPER PARAMETER ###############
# train #
IMAGE_SIZE = 224 # 448
CELL_SIZE = 7
BOXES_PER_CELL = 2
DISP_CONSOLE = False
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0
WIDTH_MULTIPLIER = 0.5
DEPTH_MULTIPLIER = 2.0
LEARNING_RATE = 1e-3
MAX_ITER = 50000
SUMMARY_ITER = 10
SAVE_ITER = 1000
DECAY_STEPS = 1000
DECAY_RATE = 0.9
STAIRCASE = False
BATCH_SIZE = 100
L2_REG_RATE = 1e-3
BATCHNORM_DECAY_RATE = 0.9
DROPOUT_KEEP_PROB = 0.5

RESTORE = False

# test parameter
THRESHOLD = 0.2
IOU_THRESHOLD = 0.5



# IMAGE_SIZE = 224 # 448
# CELL_SIZE = 7
# BOXES_PER_CELL = 2
# DISP_CONSOLE = False
# OBJECT_SCALE = 1.0
# NOOBJECT_SCALE = 1.0
# CLASS_SCALE = 2.0
# COORD_SCALE = 5.0
# WIDTH_MULTIPLIER = 1.0
# LEARNING_RATE = 0.0001
# MAX_ITER = 50000
# SUMMARY_ITER = 10
# SAVE_ITER = 1000
# DECAY_STEPS = 5000
# DECAY_RATE = 0.1
# STAIRCASE = True
# BATCH_SIZE = 50
# L2_REG_RATE = 1e-3
# BATCHNORM_DECAY_RATE = 0.9
# DROPOUT_KEEP_PROB = 0.5