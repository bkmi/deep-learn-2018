#!/usr/bin/env bash

EPOCHS=700
SAVE=true
python train.py --gantype iwgan --savepics $SAVE --savemodels $SAVE --epochs $EPOCHS
mv save save_iwgan
python train.py --gantype wgan --savepics $SAVE --savemodels $SAVE --epochs $EPOCHS
mv save save_wgan
python train.py --gantype dcgan --savepics $SAVE --savemodels $SAVE --epochs $EPOCHS
mv save save_dcgan
