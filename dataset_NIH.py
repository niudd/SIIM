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
from augmentation import do_augmentation


class SIIMDataset(Dataset):
    def __init__(self, img_id_list, IMG_SIZE, mode='train', augmentation=False):
        self.img_id_list = img_id_list
        self.IMG_SIZE = IMG_SIZE
        self.mode = mode
        self.augmentation = augmentation
        if self.mode=='train':
            #self.path = 'data/raw/NIH/NIH external data/'
            self.label_df = pd.read_csv('data/raw/NIH/NIH external data/label_df_pneumothorax.csv')
        elif self.mode=='test':
            self.path = 'data/processed/test/'
    
    def __getitem__(self, idx):
        if self.mode=='train':
            img_id = self.img_id_list[idx]
            
            img_path = glob.glob('data/raw/NIH/NIH external data/images*/'+img_id)[0]#00000003_000.png
            
            img = plt.imread(img_path)
            if len(img.shape)==3:#in this dataset, some images has 3/4 channels
                img = img[:,:,0]
            width, height = img.shape
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = np.expand_dims(img, 0)

            label = self.label_df.loc[self.label_df.img_id==img_id, 'has_pneumothorax'].values[0]
            label = np.array([label])
            
            ##augmentation
            if self.augmentation:
                img = do_augmentation(img, mask=None)
            return img, label

        elif self.mode=='test':
            img_id = self.img_id_list[idx]
            img_path = self.path + '%s.png'%img_id
            img = plt.imread(img_path)
            width, height = img.shape
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = np.expand_dims(img, 0)
            return img
    
    def __len__(self):
        return len(self.img_id_list)

def prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, IMG_SIZE=512, debug=False, use_sampler=True):
    #stratified split dataset by label
    label_df = pd.read_csv('data/raw/NIH/NIH external data/label_df_pneumothorax.csv')
    
    train_fname_list = label_df['img_id'].to_list()
    label = label_df['has_pneumothorax'].to_list()
    train_fnames, valid_fnames = train_test_split(train_fname_list, test_size=0.1, 
                                                  stratify=label, random_state=SEED)

    #debug mode
    if debug:
        train_fnames = np.random.choice(train_fnames, 9000, replace=True).tolist()
        valid_fnames = np.random.choice(valid_fnames, 1000, replace=True).tolist()
    
    #sampler with weights, and sample 10% images to train (~1w of 11w total)
    if use_sampler:
        class_weights = make_class_weights(train_fnames, label_df)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=class_weights, 
                        num_samples=int(round(len(class_weights)*0.1)), replacement=False)#total 11w images
        valid_fnames = make_validset(valid_fnames, label_df, SEED, percentage=0.1)

        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    print('Count of trainset (for training): ', len(train_fnames))
    print('Count of validset (for training): ', len(valid_fnames))
    
    ## build pytorch dataset and dataloader
    train_ds = SIIMDataset(train_fnames, IMG_SIZE, mode='train', augmentation=True)
    val_ds = SIIMDataset(valid_fnames, IMG_SIZE, mode='train', augmentation=False)
    #print(len(train_ds.fname_list), len(val_ds.fname_list))
    
    train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle, #True,
            sampler=train_sampler,
            num_workers=NUM_WORKERS,
            drop_last=True
        )
    val_dl = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            #sampler=val_sampler,
            num_workers=NUM_WORKERS,
            drop_last=True
        )
    
    return train_dl, val_dl

def prepare_testset(BATCH_SIZE, NUM_WORKERS, IMG_SIZE=512):
    #sub = pd.read_csv('data/raw/sample_submission.csv')
    #test_fnames = sub.ImageId.tolist()
    test_fnames = [f.split('/')[-1][:-4] for f in glob.glob('data/processed/test/*')]
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

def make_class_weights(img_id_list, label_df):
    """
    competition trainset has_pneumothorax: 
    0    0.716284
    1    0.283716
    
    NIH full dataset has_pneumothorax:
    0    0.952711
    1    0.047289    
    """
    label_df = label_df.set_index('img_id')
    weights = []
    for img_id in img_id_list:
        label = label_df.loc[img_id, 'has_pneumothorax']
        if label==1:
            w = 6.0
        elif label==0:
            w = 0.75
        weights.append(w)
    return weights

# build a fix validset
def make_validset(valid_fnames, label_df, SEED, percentage=0.1):
    label_df = label_df.set_index('img_id')
    #len(label_df.index)
    n_total = int(round(len(valid_fnames) * percentage))
    n_pos = int(round(n_total * 0.283716))
    n_neg = int(n_total - n_pos)
    #n_pos, n_neg
    np.random.seed(SEED)
    val_pos = np.random.choice(label_df.groupby('has_pneumothorax').get_group(1).index, n_pos, replace=False).tolist()
    val_neg = np.random.choice(label_df.groupby('has_pneumothorax').get_group(0).index, n_neg, replace=False).tolist()
    val_pos.extend(val_neg)
    return val_pos
