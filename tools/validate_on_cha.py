#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:qinxiaoran
@file:validate_on_lfw.py
@time:2018/01/13

validate on lfw for caffe model
"""
from __future__ import division, print_function, absolute_import

import _init_paths
import caffe
import os
import os.path as osp
import sys
import argparse
import numpy as np
import cv2
import facenet
import lfw
import characters

def im_list_to_blob(ims):
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(args, im):
    target_size = args.image_size
    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return im


def main(args):
    # Read the file containing the pairs used for testing
    pairs = characters.read_pairs(os.path.expanduser(args.lfw_pairs))
    # Get the paths for the corresponding images
    paths, actual_issame = characters.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
    
    rootdir = '/home/qinxiaoran/project/triplet'
    prototxt = osp.join(rootdir, 'models/deploy.prototxt')
    #caffemodel = osp.join(rootdir, 'output/casia_webface_triplet_nofix_hard/vgg_casia_triplet_iter_80000.caffemodel')
    caffemodel = osp.join(rootdir, 'output/comic_characters_triplet_contriblossm/vgg_cha_triplet_iter_80000.caffemodel')

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print('Loaded network {:s}'.format(caffemodel))
    
    print('Runnning forward pass on comic images')
    emb_array = []
    nrof_images = len(paths)
    for i in range(nrof_images):
        im = cv2.imread(paths[i])
        im = prep_im_for_blob(args, im)
        blob = im_list_to_blob([im])
        forward_kwargs = {'data': blob.astype(np.float32, copy=False)}
        res = net.forward(**forward_kwargs)
        res = res['norm2'][0]
        res = res.tolist()
        emb_array.append(res)
    
    emb_array = np.array(emb_array)
    print('num of embeddings: ', len(emb_array))
    tpr, fpr, accuracy, val, val_std, far = characters.evaluate(emb_array, actual_issame, nrof_folds=args.lfw_nrof_folds)
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu_id', type=int, help='GPU device id to use [0]', default=2)
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.', default='/home/qinxiaoran/project/triplet/data/comic_characters/characters_frcrop_182')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=224)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='/home/qinxiaoran/project/triplet/data/comic_characters/validation500_pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))