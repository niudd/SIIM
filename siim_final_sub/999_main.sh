
../anaconda3/bin/python3 000_preprocess_testset.py

../anaconda3/bin/python3 100_unet.py --SEED=3456 --IMG_SIZE=1024 --EMPTY_THRESHOLD=6100 --MASK_THRESHOLD=0.18 --checkpoint_path=checkpoint/UNetResNet34_1024_v1_seed3456.pth.tar --sub_fname=unet_1024_seed3456_tta_v1_6100_018.csv.gz

../anaconda3/bin/python3 100_unet.py --SEED=1234 --IMG_SIZE=1024 --EMPTY_THRESHOLD=5300 --MASK_THRESHOLD=0.22 --checkpoint_path=checkpoint/UNetResNet34_1024_v1_seed1234.pth.tar --sub_fname=unet_1024_seed1234_tta_v1_5300_022.csv.gz

../anaconda3/bin/python3 100_unet.py --SEED=2345 --IMG_SIZE=1024 --EMPTY_THRESHOLD=6000 --MASK_THRESHOLD=0.18 --checkpoint_path=checkpoint/UNetResNet34_1024_v1_seed2345.pth.tar --sub_fname=unet_1024_seed2345_tta_v1_6000_018.csv.gz

../anaconda3/bin/python3 100_unet.py --SEED=1234 --IMG_SIZE=768 --EMPTY_THRESHOLD=2900 --MASK_THRESHOLD=0.18 --checkpoint_path=checkpoint/UNetResNet34_768_v1_seed1234.pth.tar --sub_fname=unet_768_seed1234_tta_v1_2900_018.csv.gz

../anaconda3/bin/python3 101_deeplabv3plus.py --SEED=9012 --IMG_SIZE=1024 --EMPTY_THRESHOLD=6000 --MASK_THRESHOLD=0.18 --checkpoint_path=checkpoint/deeplabv3plus_resnet_1024_v2_seed9012.pth.tar --sub_fname=deeplabv3plus_1024_seed9012_tta_v2_6000_018.csv.gz

../anaconda3/bin/python3 101_deeplabv3plus.py --SEED=5678 --IMG_SIZE=1024 --EMPTY_THRESHOLD=6000 --MASK_THRESHOLD=0.22 --checkpoint_path=checkpoint/deeplabv3plus_resnet_1024_v2_seed5678.pth.tar --sub_fname=deeplabv3plus_1024_seed5678_tta_v2_6000_022.csv.gz

../anaconda3/bin/python3 101_deeplabv3plus.py --SEED=6789 --IMG_SIZE=1024 --EMPTY_THRESHOLD=6000 --MASK_THRESHOLD=0.19 --checkpoint_path=checkpoint/deeplabv3plus_resnet_1024_v2_seed6789.pth.tar --sub_fname=deeplabv3plus_1024_seed6789_tta_v2_6000_019.csv.gz

../anaconda3/bin/python3 101_deeplabv3plus.py --SEED=1234 --IMG_SIZE=1024 --EMPTY_THRESHOLD=6000 --MASK_THRESHOLD=0.22 --checkpoint_path=checkpoint/deeplabv3plus_resnet_1024_v2_seed1234.pth.tar --sub_fname=deeplabv3plus_1024_seed1234_tta_v2_6000_022.csv.gz

../anaconda3/bin/python3 101_deeplabv3plus.py --SEED=1234 --IMG_SIZE=768 --EMPTY_THRESHOLD=2900 --MASK_THRESHOLD=0.22 --checkpoint_path=checkpoint/deeplabv3plus_resnet_768_v1_seed1234.pth.tar --sub_fname=deeplabv3plus_768_seed1234_tta_v1_2900_022.csv.gz

../anaconda3/bin/python3 101_deeplabv3plus.py --SEED=3456 --IMG_SIZE=1024 --EMPTY_THRESHOLD=6300 --MASK_THRESHOLD=0.23 --checkpoint_path=checkpoint/deeplabv3plus_resnet_1024_v2_seed3456.pth.tar --sub_fname=deeplabv3plus_1024_seed3456_tta_v2_6300_023.csv.gz

../anaconda3/bin/python3 101_deeplabv3plus.py --SEED=2345 --IMG_SIZE=1024 --EMPTY_THRESHOLD=5700 --MASK_THRESHOLD=0.18 --checkpoint_path=checkpoint/deeplabv3plus_resnet_1024_v2_seed2345.pth.tar --sub_fname=deeplabv3plus_1024_seed2345_tta_v2_5700_018.csv.gz

../anaconda3/bin/python3 200_ensemble.py --model=0
../anaconda3/bin/python3 200_ensemble.py --model=1


