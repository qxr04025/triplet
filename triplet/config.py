import os

# Training image data path
#IMAGEPATH = '/home/qinxiaoran/project/triplet/data/CASIA-Webface/CASIA-WebFace-MTCNNcrop_182'
IMAGEPATH = '/home/qinxiaoran/project/triplet/data/comic_characters/characters_frcrop_182_train1000'

# LFW image data path
#LFW_IMAGEPATH = 'data/LFW/lfw-deepfunneled/'

# Path to caffe directory
CAFFEPATH = '/home/qinxiaoran/project/face-py-faster-rcnn/caffe-fast-rcnn'

# Snapshot iteration
SNAPSHOT_ITERS = 10000

# Max training iteration
#8epoch for casia softmax
MAX_ITERS = 80000
#MAX_ITERS = 20000

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

#  USE contrib-loss negative mining during training?
CONTRIB_LOSS = True

# Number of samples of each identity in a minibatch
CUT_SIZE = 5
