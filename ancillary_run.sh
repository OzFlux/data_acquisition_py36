#!/bin/bash

# Run the AWS update job
/mnt/miniconda2/envs/py36/bin/python /mnt/OzFlux/test_code/data_acquisition_py36/fetch_bom_data_new.py

# Run the ACCESS update job
/mnt/miniconda2/envs/py36/bin/python /mnt/OzFlux/test_code/data_acquisition_py36/fetch_access_3.py
