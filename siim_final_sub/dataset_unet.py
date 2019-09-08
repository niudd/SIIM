import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import collections
from tqdm import tqdm_notebook, tqdm
import os
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import random
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from mask_functions import rle2mask
from augmentation import do_augmentation


class SIIMDataset(Dataset):
    def __init__(self, img_id_list, IMG_SIZE, mode='train', augmentation=False):
        self.img_id_list = img_id_list
        self.IMG_SIZE = IMG_SIZE
        self.mode = mode
        self.augmentation = augmentation
        if self.mode=='train':
            pass
        elif self.mode=='test':
            self.path = 'data/test/'
            self.data = self.img_id_list#for __len__
    
    def __getitem__(self, idx):
        if self.mode=='train':
            item = self.data[idx]
            img_path = self.path + '%s.png'%item['img_id']
            img = plt.imread(img_path)
            width, height = img.shape
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = np.expand_dims(img, 0)

            cnt_masks = item['cnt_masks']
            masks_in_rle = item['masks']
            if cnt_masks==1:
                mask = rle2mask(masks_in_rle[0], width, height).T
            elif cnt_masks>1: #v1: just simply merge overlapping masks to get union of masks
                masks = []
                for mask_in_rle in masks_in_rle:
                    mask = rle2mask(mask_in_rle, width, height).T
                    masks.append(mask)
                mask = (np.array(masks).sum(axis=0)>=1).astype(np.int)#mask as 1 if at least one of the mask is 1
            else:
                mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE))
            mask = cv2.resize(mask.astype(np.float), (self.IMG_SIZE, self.IMG_SIZE))
            mask = np.expand_dims(mask, 0)
            ##augmentation
            if self.augmentation:
                img, mask = do_augmentation(img, mask)
            return img, mask
        elif self.mode=='test':
            img_id = self.img_id_list[idx]
            img_path = self.path + '%s.png'%img_id
            img = plt.imread(img_path)
            width, height = img.shape
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = np.expand_dims(img, 0)
            return img
    
    def __len__(self):
        return len(self.data)


def prepare_testset(BATCH_SIZE, NUM_WORKERS, IMG_SIZE=512):
    #sub = pd.read_csv('data/raw/sample_submission.csv')
    #test_fnames = sub.ImageId.tolist()
    test_fnames = [f.split('/')[-1][:-4] for f in glob.glob('data/test/*')]
    print('in dataset_unet: ', len(test_fnames))
    test_ds = SIIMDataset(test_fnames, IMG_SIZE, mode='test', augmentation=False)
    #print(len(train_ds.fname_list), len(val_ds.fname_list))
    test_dl = DataLoader(
                        test_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        #sampler=sampler,
                        num_workers=NUM_WORKERS,
                        drop_last=False
                    )
    return test_dl

