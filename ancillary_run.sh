#!/bin/bash

# Run the AWS update job
/mnt/miniconda2/envs/py36/bin/python /mnt/OzFlux/test_code/data_acquisition_py36/fetch_bom_data.py

# Run the ACCESS update job 
/mnt/miniconda2/envs/py36/bin/python /mnt/OzFlux/test_code/data_acquisition_py36/fetch_access_data.py

# Run the ACCESS write job
/mnt/miniconda2/envs/py36/bin/python /mnt/OzFlux/test_code/data_acquisition_py36/convert_access.py
