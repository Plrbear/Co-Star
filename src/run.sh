#!/bin/bash

export HYDRA_FULL_ERROR=1




python train.py \
  extra.temperature=2 \
  extra.temperature_clip_reliability=0.5 \
  extra.temperature_clip_collaborative=0.5 \
  extra.scale=6 \
  extra.prob_invert=0.25 \
  run_name=K_M_scron_on \
  extra.ema_decay=0.999 \
  extra.scr=false \
  trainer.max_epochs=40\
  extra.min_weight=0\
  extra.max_weight=1 \
  extra.mu_temp=1 \
  extra.beta_temp=0.009 \
  extra.imp_fac=0.2

