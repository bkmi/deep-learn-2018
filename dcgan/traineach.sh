#!/usr/bin/env bash

EPOCHS=1
SAVE=true
python train.py --gantype dcgan --savepics $SAVE --savemodels $SAVE --epochs $EPOCHS
mv save save_dcgan
python train.py --gantype wgan --savepics $SAVE --savemodels $SAVE --epochs $EPOCHS
mv save save_dcgan
python train.py --gantype iwgan --savepics $SAVE --savemodels $SAVE --epochs $EPOCHS
mv save save_dcgan