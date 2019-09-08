## Experiments
- predict mask and classify zero-mask-nonzero-mask in one model training, worse❌
- image size 512, gradient accumulation, sync batchnorm for small batch size(=8)...train a network on 512x512 for around 40 epochs.✅✅✅✅
- 2-step-schema: classify zero-mask-nonzero-mask with all samples, then train mask model with all nonzero mask samples. first step not working, zero-nonzero-mask performance AUC score worse than unet mask model's sum-of-pixels
- ensemble 5 seeds, ensemble the model logits by (sigmoid(logit0) + ... + sigmoid(logit4)) / 5, (5 seeds predict on the same validset), and use that for thresholds searching on a single fold (seed=3456). local CV=0.88+, LB=0.85
- multitask of naively adding classification path onto "center block". no improvement on lb
- DeepSupervision architecture based on my unet, seems no significant improvements
- deeplab v3+, LB score similar to UNetResNet34 (best model)
- deeplab v3+, with 768 size, current best model
- use NIH ~11w data (clf only) to train has_pneumothorax classifier (unet), step 2 to use this model as pretrained (encoder) to train mask model (decoder) on ~1w competition data



## TODO
- Unet and its extensions, e.g. attention Unet...
- try deeplab v3+, modify it with DeepSupervision❓
- add more TTA methods, e.g. multi-scale inputs, vertical flip
- some hard example mining? (false negative)

## Knowledge
- threshold search improves score
- [split mask into multiple instances improves score❌, 07-13 official announce metircs changed, now it is semantic segmentation, one mask per image. As result, all splited-masks submission are worse. DO NOT SPLIT MASK !!!](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/99231#latest-574195)
- IMG_SIZE bigger helps, 512: LB=0.8265 from 256:LB=0.81+
- Weird: seed_everything decrease GPU usage?
- train deeplab on 512, then finetune 20 epochs on 1024, improves LB 
- TTA improves (a little bit)
- WOW, use voting ensemble improves; different networks ensemble improves significantly (e.g. unet+deeplabv3p)
- GT=Empty mask, prediction mostly empty; GT=Non-Empty but very small mask, some bad case predicting Empty mask; i.e. false negative is a problem
- training could accidentally stop or save best epoch too early (20~30 epochs), we should set minimum epochs of 40

## model checkpoints
- UNetResNet34_512_v1_seedxxxx: BCE loss, current best model
- UNetResNet34_512_v2_seedxxxx: bce + dice
- UNetResNet34_512_v3_seedxxxx: finetune with bce + dice
- UNetResNet34_512_v4_seedxxxx: train on nonzero-mask image
- UNetResNet34_512_v5_seedxxxx: symmetric_lovasz loss from kaggler's / Focal loss
- UNetResNet34_512_v6_seedxxxx: Heng's weighted bce loss and so on...training is difficult
- multitask_UNetResNet34_512_v1_seedxxxx: add classification task, LB similar to best model (maybe lower)
- multitask_UNetResNet34_512_v2_seedxxxx: similar to v1, change loss: mask loss only on nonempty images. modify metrics to dice_multitask
- deep_supervision_UNetResNet34_512_v1_seedxxxx: 3 BCE losses, LB=0.8389, predict 22% non-empty-mask on validset (seems close to GT)
- deep_supervision_UNetResNet34_512_v2_seedxxxx: finetune with symmetric_lovasz loss. training process not working
- deeplabv3plus_resnet_512_v1_seedxxxx: LB similar to UNetResNet34_512_v1_seedxxxx
- deeplabv3plus_resnet_512_v2_seedxxxx: fintune on v1, with weighted bce from Heng. training val dice not decreasing
- deeplabv3plus_resnet_1024_v1_seedxxxx: warm start from 512 checkpoint, stage 2 --> fintune on 1024x1024, LB=.8538/.8503
- deeplabv3plus_resnet_768_v1_seedxxxx: train directly on 768 size, LB=0.8551/0.8554/0.8605, best single model till now
- UNetResNet34_768_v1_seedxxxx: train on 768 size, LB improves than 512, LB=0.8557/0.8530/0.8592
- deeplabv3plus_supervision_resnet_1024_v1_seedxxxx: deeplabv3plus add deep supervision, LB=0.8561 < 0.8614

## current best
- deeplabv3plus_resnet_1024_v2_seedxxxx: train 1024 size from scratch, LB improves than 768, LB=0.8614/0.8597/0.8603/0.8632, seed4567=0.8548 maybe train again?
- UNetResNet34_1024_v1_seedxxxx: train on 1024 from scratch, LB=0.8601/0.8592/0.8609, Unet tempts to predict more masks? (sub file size larger than deeplab's)


deeplabv3+ from seed=5678, unet from seed=3456, adding train at least 40 epochs trick, scores good







