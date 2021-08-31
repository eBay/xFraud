# !/bin/bash
# run this script under project's dir

conda env create -f environment_20210301.yml
conda activate xfraud_test

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch -y
conda install ignite -c pytorch -y

pip install --no-index \
    torch-scatter torch-sparse torch-cluster torch-spline-conv -f \
    https://pytorch-geometric.com/whl/torch-1.8.1+cu101.html

pip install torch-geometric==1.7.2