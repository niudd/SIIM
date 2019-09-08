import numpy as np
import pandas as pd
import os
from glob import glob
import sys
#import skimage.measure
import PIL
from tqdm import tqdm, tqdm_notebook

from mask_functions import rle2mask, mask2rle
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='====Model Parameters====')
parser.add_argument('--model', type=int, default=0)

params = parser.parse_args()
model = params.model

if model==0:
    use_subs = ['submission/deeplabv3plus_1024_seed5678_tta_v2_6000_022.csv.gz',
                'submission/deeplabv3plus_1024_seed1234_tta_v2_6000_022.csv.gz',
                'submission/deeplabv3plus_768_seed1234_tta_v1_2900_022.csv.gz',
                'submission/unet_1024_seed3456_tta_v1_6100_018.csv.gz',
                'submission/deeplabv3plus_1024_seed3456_tta_v2_6300_023.csv.gz',
                'submission/unet_1024_seed1234_tta_v1_5300_022.csv.gz',
                'submission/deeplabv3plus_1024_seed2345_tta_v2_5700_018.csv.gz',
                'submission/unet_1024_seed2345_tta_v1_6000_018.csv.gz',
                'submission/unet_768_seed1234_tta_v1_2900_018.csv.gz',
               ] #LB=0.8711, min_solutions=4

    print('use how many subs: ', len(use_subs))
    df_sub_list = [pd.read_csv(f) for f in use_subs]
    print(use_subs)

    # create a list of unique image IDs
    for _i in range(len(df_sub_list)):
        iid_list = df_sub_list[_i]["ImageId"].unique()
        print(f"{len(iid_list)} unique image IDs.")

    ## Create average prediction mask for each image ##
    # set here the threshold for the final mask
    # min_solutions is the minimum number of times that a pixel has to be positive in order to be included in the final mask
    min_solutions = 4 # if avg_mask>=min_solutions, then predict a pixel=1
    assert (min_solutions >= 1 and min_solutions <= len(df_sub_list)), \
        "min_solutions has to be a number between 1 and the number of submission files"
    print('min_solutions: ', min_solutions)


    # create empty final dataframe
    df_avg_sub = pd.DataFrame(columns=["ImageId", "EncodedPixels"])
    df_avg_sub_idx = 0 # counter for the index of the final dataframe

    # iterate over image IDs
    for idx, iid in enumerate(iid_list):
        # initialize prediction mask
        avg_mask = np.zeros((1024,1024))
        # iterate over prediction dataframes
        for df_sub in df_sub_list:
            # extract rles for each image ID and submission dataframe
            rles = df_sub.loc[df_sub["ImageId"]==iid, "EncodedPixels"]
            # iterate over rles
            for rle in rles:
                # if rle is not -1, build prediction mask and add to average mask
                if "-1" not in str(rle):
                    avg_mask += rle2mask(rle, 1024, 1024) / float(len(df_sub_list))
        # threshold the average mask
        pred_mask = (avg_mask >= (min_solutions / float(len(df_sub_list)))).astype("uint8")
        # transform to rle
        if pred_mask.sum() > 0:
            im = PIL.Image.fromarray((pred_mask*255).astype(np.uint8)).resize((1024,1024))
            im = np.asarray(im)
            rle = mask2rle(im, 1024, 1024)
        else:
            rle = "-1"
        # add a row in the final dataframe
        df_avg_sub.loc[df_avg_sub_idx] = [iid, rle]
        df_avg_sub_idx += 1 # increment index
        #if idx>10:
        #    break

    df_avg_sub.to_csv('submission/submission0.csv.gz', index=False, compression='gzip')

elif model==1:
    use_subs = ['submission/deeplabv3plus_1024_seed9012_tta_v2_6000_018.csv.gz',
                'submission/deeplabv3plus_1024_seed5678_tta_v2_6000_022.csv.gz',
                'submission/deeplabv3plus_1024_seed6789_tta_v2_6000_019.csv.gz',
                'submission/deeplabv3plus_1024_seed1234_tta_v2_6000_022.csv.gz',
                'submission/deeplabv3plus_768_seed1234_tta_v1_2900_022.csv.gz',
                'submission/unet_1024_seed3456_tta_v1_6100_018.csv.gz',
                'submission/unet_1024_seed1234_tta_v1_5300_022.csv.gz',
                'submission/unet_1024_seed2345_tta_v1_6000_018.csv.gz',
                'submission/unet_768_seed1234_tta_v1_2900_018.csv.gz',
               ] #LB=0.8708, min_solutions=5

    print('use how many subs: ', len(use_subs))
    df_sub_list = [pd.read_csv(f) for f in use_subs]
    print(use_subs)

    # create a list of unique image IDs
    for _i in range(len(df_sub_list)):
        iid_list = df_sub_list[_i]["ImageId"].unique()
        print(f"{len(iid_list)} unique image IDs.")

    ## Create average prediction mask for each image ##
    # set here the threshold for the final mask
    # min_solutions is the minimum number of times that a pixel has to be positive in order to be included in the final mask
    min_solutions = 5 # if avg_mask>=min_solutions, then predict a pixel=1
    assert (min_solutions >= 1 and min_solutions <= len(df_sub_list)), \
        "min_solutions has to be a number between 1 and the number of submission files"
    print('min_solutions: ', min_solutions)


    # create empty final dataframe
    df_avg_sub = pd.DataFrame(columns=["ImageId", "EncodedPixels"])
    df_avg_sub_idx = 0 # counter for the index of the final dataframe

    # iterate over image IDs
    for idx, iid in enumerate(iid_list):
        # initialize prediction mask
        avg_mask = np.zeros((1024,1024))
        # iterate over prediction dataframes
        for df_sub in df_sub_list:
            # extract rles for each image ID and submission dataframe
            rles = df_sub.loc[df_sub["ImageId"]==iid, "EncodedPixels"]
            # iterate over rles
            for rle in rles:
                # if rle is not -1, build prediction mask and add to average mask
                if "-1" not in str(rle):
                    avg_mask += rle2mask(rle, 1024, 1024) / float(len(df_sub_list))
        # threshold the average mask
        pred_mask = (avg_mask >= (min_solutions / float(len(df_sub_list)))).astype("uint8")
        # transform to rle
        if pred_mask.sum() > 0:
            im = PIL.Image.fromarray((pred_mask*255).astype(np.uint8)).resize((1024,1024))
            im = np.asarray(im)
            rle = mask2rle(im, 1024, 1024)
        else:
            rle = "-1"
        # add a row in the final dataframe
        df_avg_sub.loc[df_avg_sub_idx] = [iid, rle]
        df_avg_sub_idx += 1 # increment index
        #if idx>10:
        #    break

    df_avg_sub.to_csv('submission/submission1.csv.gz', index=False, compression='gzip')

elif model==2:
    use_subs = ['submission/deeplabv3plus_1024_seed9012_tta_v2_6000_018.csv.gz',
                'submission/deeplabv3plus_1024_seed5678_tta_v2_6000_022.csv.gz',
                'submission/deeplabv3plus_1024_seed6789_tta_v2_6000_019.csv.gz',
                'submission/deeplabv3plus_1024_seed1234_tta_v2_6000_022.csv.gz',
                'submission/deeplabv3plus_768_seed1234_tta_v1_2900_022.csv.gz',
                'submission/deeplabv3plus_1024_seed3456_tta_v2_6300_023.csv.gz',
                'submission/deeplabv3plus_1024_seed2345_tta_v2_5700_018.csv.gz',
                'submission/unet_1024_seed3456_tta_v1_6100_018.csv.gz',
                'submission/unet_1024_seed1234_tta_v1_5300_022.csv.gz',
                'submission/unet_1024_seed2345_tta_v1_6000_018.csv.gz',
                'submission/unet_768_seed1234_tta_v1_2900_018.csv.gz',
               ] #LB=0.8706, min_solutions=5

    print('use how many subs: ', len(use_subs))
    df_sub_list = [pd.read_csv(f) for f in use_subs]
    print(use_subs)

    # create a list of unique image IDs
    for _i in range(len(df_sub_list)):
        iid_list = df_sub_list[_i]["ImageId"].unique()
        print(f"{len(iid_list)} unique image IDs.")

    ## Create average prediction mask for each image ##
    # set here the threshold for the final mask
    # min_solutions is the minimum number of times that a pixel has to be positive in order to be included in the final mask
    min_solutions = 5 # if avg_mask>=min_solutions, then predict a pixel=1
    assert (min_solutions >= 1 and min_solutions <= len(df_sub_list)), \
        "min_solutions has to be a number between 1 and the number of submission files"
    print('min_solutions: ', min_solutions)


    # create empty final dataframe
    df_avg_sub = pd.DataFrame(columns=["ImageId", "EncodedPixels"])
    df_avg_sub_idx = 0 # counter for the index of the final dataframe

    # iterate over image IDs
    for idx, iid in enumerate(iid_list):
        # initialize prediction mask
        avg_mask = np.zeros((1024,1024))
        # iterate over prediction dataframes
        for df_sub in df_sub_list:
            # extract rles for each image ID and submission dataframe
            rles = df_sub.loc[df_sub["ImageId"]==iid, "EncodedPixels"]
            # iterate over rles
            for rle in rles:
                # if rle is not -1, build prediction mask and add to average mask
                if "-1" not in str(rle):
                    avg_mask += rle2mask(rle, 1024, 1024) / float(len(df_sub_list))
        # threshold the average mask
        pred_mask = (avg_mask >= (min_solutions / float(len(df_sub_list)))).astype("uint8")
        # transform to rle
        if pred_mask.sum() > 0:
            im = PIL.Image.fromarray((pred_mask*255).astype(np.uint8)).resize((1024,1024))
            im = np.asarray(im)
            rle = mask2rle(im, 1024, 1024)
        else:
            rle = "-1"
        # add a row in the final dataframe
        df_avg_sub.loc[df_avg_sub_idx] = [iid, rle]
        df_avg_sub_idx += 1 # increment index
        #if idx>10:
        #    break

    df_avg_sub.to_csv('submission/submission2.csv.gz', index=False, compression='gzip')