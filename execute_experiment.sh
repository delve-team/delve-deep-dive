#!/usr/bin/env bash
echo $2
python run.py $@
python probe_data_collecter $@
python train_probes --config $2 -mp 8 -f ./latent_datasets/