import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from tqdm import tqdm, tqdm_notebook
import pickle
import os
import logging
import time
import gc
import glob
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import save_checkpoint, load_checkpoint, set_logger
from gpu_utils import set_n_get_device

from model.model_unet2 import UNetResNet34, predict_proba
from dataset_unet import prepare_testset

import PIL
from mask_functions import mask2rle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
    return np.log(x / (1-x))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #tf.set_random_seed(seed)

######### Config #########
# variables
parser = argparse.ArgumentParser(description='====Model Parameters====')
parser.add_argument('--SEED', type=int, default=1234)
parser.add_argument('--IMG_SIZE', type=int, default=1024)
parser.add_argument('--EMPTY_THRESHOLD', type=int, default=6000)
parser.add_argument('--MASK_THRESHOLD', type=float, default=0.18)
parser.add_argument('--checkpoint_path', type=str, default='checkpoint/UNetResNet34_1024_v1_seed9012/best.pth.tar')
parser.add_argument('--sub_fname', type=str, default='')
params = parser.parse_args()
SEED = params.SEED
IMG_SIZE = params.IMG_SIZE
EMPTY_THRESHOLD = params.EMPTY_THRESHOLD
MASK_THRESHOLD = params.MASK_THRESHOLD
checkpoint_path = params.checkpoint_path
sub_fname = params.sub_fname
print('######### Config #########')
print('SEED=', SEED)
print('IMG_SIZE=', IMG_SIZE)
print('EMPTY_THRESHOLD=', EMPTY_THRESHOLD)
print('MASK_THRESHOLD=', MASK_THRESHOLD)
print('checkpoint_path=', checkpoint_path)
print('sub_fname=', sub_fname)
print('\n')

# constants
MODEL = 'UNetResNet34'
print('====MODEL ACHITECTURE: %s===='%MODEL)
device = set_n_get_device("0,1", data_device_id="cuda:0")
multi_gpu = [0,1] #None#[0, 1]#use 2 gpus
debug = False # if True, load 100 samples
BATCH_SIZE = 8
NUM_WORKERS = 24

seed_everything(SEED)

######### Run prediction #########
net = UNetResNet34(debug=False).cuda(device=device)
net, _ = load_checkpoint(checkpoint_path, net)
if multi_gpu is not None:
    net = nn.DataParallel(net, device_ids=multi_gpu)

test_fnames = [f.split('/')[-1][:-4] for f in glob.glob('data/test/*')]
print(len(test_fnames), test_fnames[0])

test_dl = prepare_testset(BATCH_SIZE, NUM_WORKERS, IMG_SIZE)
preds_test = predict_proba(net, test_dl, device, multi_gpu=multi_gpu, mode='test', tta=True)
print(preds_test.shape)


def predict_mask(logit, EMPTY_THRESHOLD, MASK_THRESHOLD):
    """Transform each prediction into mask.
    input shape: (256, 256)
    """
    #pred mask 0-1 pixel-wise
    #n = logit.shape[0]
    IMG_SIZE = logit.shape[-1] #256
    #EMPTY_THRESHOLD = 100.0*(IMG_SIZE/128.0)**2 #count of predicted mask pixles<threshold, predict as empty mask image
    #MASK_THRESHOLD = 0.22
    #logit = torch.sigmoid(torch.from_numpy(logit)).view(n, -1)
    #pred = (logit>MASK_THRESHOLD).long()
    #pred[pred.sum(dim=1) < EMPTY_THRESHOLD, ] = 0 #bug here, found it, the bug is input shape is (256, 256) not (16,256,256)
    logit = sigmoid(logit)#.reshape(n, -1)
    pred = (logit>MASK_THRESHOLD).astype(np.int)
    if pred.sum() < EMPTY_THRESHOLD:
        return np.zeros(pred.shape).astype(np.int)
    else:
        return pred

######### Build submission #########
# Generate rle encodings (images are first converted to the original size)
rles = []
for p in preds_test:#p is logit from model
    pred_mask = predict_mask(p, EMPTY_THRESHOLD, MASK_THRESHOLD)
    if pred_mask.sum()>0:#predicted non-empty mask
        im = PIL.Image.fromarray((pred_mask.T*255).astype(np.uint8)).resize((1024,1024))
        im = np.asarray(im)
        rles.append(mask2rle(im, 1024, 1024))
    else: rles.append('-1')
    
sub_df = pd.DataFrame({'ImageId': test_fnames, 'EncodedPixels': rles})
print(len(sub_df.index))

sub_df.to_csv('submission/'+sub_fname, index=False, compression='gzip')


print((sub_df.EncodedPixels!='-1').mean())
