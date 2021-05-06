conda env create -f environment_20210301.yml
conda activate eth

conda install pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch -y
pip install --no-index \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-geometric==1.5.0
pip install pytorch-ignite==0.4.2