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
torch.multiprocessing.set_sharing_strategy('file_system')


class SIIMDataset(Dataset):
    def __init__(self, img_id_list, IMG_SIZE, mode='train'):
        """
        TODO: how to use empty mask images ???
        Is empty mask images used in maskRCNN ?
        
        version 1: Skip empty mask images, use non-empty-mask images to train maskRCNN, modify in prepare_trainset() !!!
        """
        self.img_id_list = img_id_list
        self.IMG_SIZE = IMG_SIZE
        self.mode = mode
        if self.mode=='train':
            #read and transform mask data
            self.mask_data = mask2data()
            self.data = [item for item in self.mask_data if item['img_id'] in img_id_list]
            self.path = 'data/processed/train/'
    
    def __getitem__(self, idx):
        item = self.data[idx]
        ## 1. the image part ##
        img_id = item['img_id']
        img_path = self.path + '%s.png'%img_id
        img = plt.imread(img_path)
        width, height = img.shape
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        ## 2. the target part ##
        cnt_masks = item['cnt_masks']
        masks_in_rle = item['masks']
        if cnt_masks==1:
            mask = rle2mask(masks_in_rle[0], width, height).T
            mask = cv2.resize(mask.astype(np.float), (self.IMG_SIZE, self.IMG_SIZE))
            masks = [mask]
            boxes = [self._get_box(mask)]#add bounding boxes, for instance segmentation task
        elif cnt_masks>1:
            masks = []
            boxes = []
            for mask_in_rle in masks_in_rle:
                mask = rle2mask(mask_in_rle, width, height).T
                mask = cv2.resize(mask.astype(np.float), (self.IMG_SIZE, self.IMG_SIZE))
                masks.append(mask)
                box = self._get_box(mask)
                boxes.append(box)
        else:
            masks = [[]]#[np.zeros((self.IMG_SIZE, self.IMG_SIZE))]
            boxes = [[]]
        masks = np.array(masks, dtype=np.float)
        boxes = np.array(boxes, dtype=np.float)
        labels = np.array([1]*cnt_masks, dtype=np.int64)
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except:
            area = np.array([], dtype=np.float)
        iscrowd = np.zeros((cnt_masks, ), dtype=np.int)
        ## 3. the output part ##
        target = {}
        target["boxes"] = torch.from_numpy(boxes).float()
        target["labels"] = torch.from_numpy(labels)
        target["masks"] = torch.from_numpy(masks).int()
        target["image_id"] = torch.from_numpy(np.array([idx]))#img_id
        target["area"] = torch.from_numpy(area)
        target["iscrowd"] = torch.from_numpy(iscrowd)
        return img, target
    
    def _get_box(self, mask):
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        #boxes = np.array([[xmin, ymin, xmax, ymax]], dtype=np.float)
        boxes = [xmin, ymin, xmax, ymax]
        return boxes
    
    def __len__(self):
        return len(self.data)

def mask2data():
    """
    return: [{}, {}, ...]
    each is 
    {'img_id': '1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027',
     'masks': ['891504 5 1018 8 1015 10 1013 12 1011 14 1009 16 1008 17', 
                '49820 3 1017 11 1012 13 1009 16 1007 18 1006 19 1005 20 1004 21', ...],
     'cnt_masks': 2}
    """
    #if preprocessed, load
    if os.path.exists('data/processed/train_mask_in_rle.pkl'):
        with open('data/processed/train_mask_in_rle.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    train_mask = pd.read_csv('data/raw/train-rle.csv')
    #img_id_exist_list = [f.split('/')[-1][:-4] for f in glob.glob('data/processed/train/*')]
    grp = train_mask.groupby('ImageId')#so image id is unique, some have multiple-masks to combine
    data = []
    for img_id, subdf in grp:
        masks = []
        for j in subdf.index:
            mask_in_rle = subdf.loc[j, ' EncodedPixels'].strip()
            if mask_in_rle!='-1':
                #mask = rle2mask(mask_in_rle, 1024, 1024).T
                #masks.append(mask)
                masks.append(mask_in_rle)
        #if masks!=[]:
        #    merged_mask = (np.array(masks).sum(axis=0)>=1).astype(np.int)#mask as 1 if at least one of the mask is 1
        #else:
        #    merged_mask = []
        data.append({'img_id': img_id, 'masks': masks, 'cnt_masks': len(masks)})#'merged_mask': merged_mask
    # save
    with open('data/processed/train_mask_in_rle.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data


def prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, IMG_SIZE=512, debug=False):
    #stratified split dataset by cnt_masks
    mask_data = mask2data()
    mask_data = [item for item in mask_data if item['cnt_masks']>0]#only use non-empty-mask images to train maskRCNN
    train_fname_list = [item['img_id'] for item in mask_data]
    cnt_masks = [item['cnt_masks'] if item['cnt_masks']<5 else 5 for item in mask_data]
    train_fnames, valid_fnames = train_test_split(train_fname_list, test_size=0.1, 
                                                  stratify=cnt_masks, random_state=SEED)

    #debug mode
    if debug:
        train_fnames = np.random.choice(train_fnames, 900, replace=True).tolist()
        valid_fnames = np.random.choice(valid_fnames, 100, replace=True).tolist()
    print('Count of trainset (for training): ', len(train_fnames))
    print('Count of validset (for training): ', len(valid_fnames))
    
    ## build pytorch dataset and dataloader
    train_ds = SIIMDataset(train_fnames, IMG_SIZE, mode='train')
    val_ds = SIIMDataset(valid_fnames, IMG_SIZE, mode='train')
    #print(len(train_ds.fname_list), len(val_ds.fname_list))
    train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            #sampler=sampler,
            num_workers=NUM_WORKERS,
            drop_last=True,
            collate_fn=lambda x: tuple(zip(*x))
        )
    val_dl = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            #sampler=sampler,
            num_workers=NUM_WORKERS,
            drop_last=True,
            collate_fn=lambda x: tuple(zip(*x))
        )
    
    return train_dl, val_dl


