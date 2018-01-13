import os

# Training image data path
IMAGEPATH = '/home/qinxiaoran/project/tripletloss/data/CASIA-Webface/CASIA-WebFace-MTCNNcrop_182'

# LFW image data path
#LFW_IMAGEPATH = 'data/LFW/lfw-deepfunneled/'

# Path to caffe directory
CAFFEPATH = '/home/qinxiaoran/project/face-py-faster-rcnn/caffe-fast-rcnn'

# Snapshot iteration
SNAPSHOT_ITERS = 10000

# Max training iteration
#8epoch for casia softmax
MAX_ITERS = 80000

# The number of samples in each minibatch for triplet loss
TRIPLET_BATCH_SIZE = 50

# The number of samples in each minibatch for other loss
BATCH_SIZE = 60

# If need to train tripletloss, set False when pre-train
TRIPLET_LOSS = True

# Use horizontally-flipped images during training?
FLIPPED = True

# training percentage
PERCENT = 0.8

# USE semi-hard negative mining during training?
SEMI_HARD = True

# Number of samples of each identity in a minibatch
CUT_SIZE = 5
