#!/usr/bin/env bash
/mnt/c/ProgramData/Anaconda3/python.exe receptive_field.py --config $2
/mnt/c/ProgramData/Anaconda3/python.exe run.py $@
/mnt/c/ProgramData/Anaconda3/python.exe probe_data_collecter.py $@
/mnt/c/ProgramData/Anaconda3/python.exe train_probes.py --config $2 -mp 8 -f ./latent_datasets/