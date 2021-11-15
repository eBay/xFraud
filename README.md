# Table of Contents
- [Introduction](#introduction)
- [Get Started](#get-started)
  - [1. Env setup](#1-env-setup)
  - [2. Feature store setup](#2-feature-store-setup)
  - [3. Detector: Training and testing](#3-detector-training-and-testing)
  - [4. Explainer](#4-explainer)
  - [5. Explainer quantitative analysis (top k hit rate)](#5-explainer-quantitative-analysis-top-k-hit-rate)
  - [Supplement material](#supplement-material)
    - [6. Centrality measures on 41 communities](#6-centrality-measures-on-41-communities)
    - [7. Learning the hybrid explainer](#7-learning-the-hybrid-explainer)
    - [8. Visualizing the communities](#8-visualizing-the-communities)
- [Data files](#data-files)
    - [Note](#note)
- [License Information](#license-information)
- [Reference](#reference)
  
# Introduction
This paper is accepted by VLDB 2022, and is published in Vol. 15, Issue 3.

xFraud is an explainable Fraud transaction prediction system. xFraud is composed of a predictor which learns expressive 
representations for malicious transaction detection from the heterogeneous transaction graph via a self-attentive 
heterogeneous graph neural network, and an explainer that generates meaningful and human understandable explanations 
from graphs to facilitate further process in business unit.
  
# Get Started

## 1. Env setup

Setup the python environment with `conda` and install `pytorch` and its dependencies. Notice for the `pytorch` related package, you may install the correct version to fit your `cuda` device. In the following experiment scripts, I will use the `cuda10.1` version.

```bash
bash ./scripts/install-env-publish.sh
```

#### Third Party Packages (Optional)
If you are going to try the code of `pyHGT` in this project, please run

```bash
bash ./scripts/add-submodules.sh
```

to include `pyHGT` as submodules

## 2. Feature store setup
This step is to prevent the OOM issue for loading large feature data.
LevelDB is used to store node features. <br>
For this data sample, to generate the data store is not a necessary step, as we also provide the feature store 
(`./data/feat_store_publish.db`) used in the subsequent scripts.

```bash
bash ./scripts/setup-feature-store.py
```

## 3. Detector: Training and testing 

```bash
bash ./scripts/run-detector.sh
```
Commonly used parameters are listed in the shell script.

## 4. Explainer

```bash 
python ./xfraud/run_explainer.py
```

## 5. Explainer quantitative analysis (top k hit rate)

```bash
python ./xfraud/explainer-eval-hitrate/ours.py
python ./xfraud/explainer-eval-hitrate/random-baseline.py
```

## Supplement material:
### 6. Centrality measures on 41 communities
```bash
cd ./xfraud/supplement/06Centrality_measures/
python edge_betweenness.py --type-centrality 'edge_betweenness_centrality' 
python line_graph_node_centrality.py --type-centrality 'degree'
```

### 7. Learning the hybrid explainer
```bash
cd ./xfraud/supplement/07Learning_hybrid/

# ridge
learn_equation-ridge.ipynb

# grid search with user defined A
python ours_learn-grid-A.py

# polynomial
learn_equation-polynomial.ipynb

# load the learned parameters and get the hybrid explainer weighs
python ours-learn.py
```

### 8. Visualizing the communities
```bash
cd ./xfraud/supplement/08Vis/

plot_community_prettify-hybrid.ipynb
```

# Data files
We provide a small sample of the transaction graph and features in `./data`. <br>
We also provide the sample, its annotations, and evaluation results (in `./xfraud/explainer-eval-hitrate`) we describe in 
the section about explainer. <br>
All the datafiles are described in the scripts that utilize them. 

## Note
The data we use in the paper is proprietary, i.e., 
real-world transaction records on the eBay platform. At this point we can only share a small dummy sample dataset 
to show the data structure and our workflow. In the long run, it would be possible to share the eBay-small 
dataset (desensitized transaction records) after the legal review at eBay. 


# License Information
Copyright 2020-2021 eBay Inc.

Author/Developer: Susie Xi Rao, Shuai Zhang, Zhichao Han, Zitao Zhang, Wei Min, Zhiyao Chen, Yinan Shan, Yang Zhao, 
Ce Zhang

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the 
License. You may obtain a copy of the License at <br>
https://www.apache.org/licenses/LICENSE-2.0 <br>
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an 
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the 
specific language governing permissions and limitations under the License.

# Reference
- `pyHGT` is from the original HGT implementation with `MIT` license, 
see [https://github.com/acbull/pyHGT/tree/master/pyHGT](https://github.com/acbull/pyHGT/tree/master/pyHGT). 

Other python package licenses are listed below:
```
 Name                    Version          License

 dill                    0.3.0            BSD License
 fire                    0.3.1            Apache Software License
 networkx                2.4              BSD License
 pandas                  0.24.2           BSD
 plyvel                  1.3.0            BSD License 
 py4j                    0.10.7           BSD License
 pyarrow                 0.15.1           Apache Software License
 scikit-learn            0.22.2.post1     new BSD
 scipy                   1.4.1            BSD License
 seaborn                 0.9.0            BSD License
 torch                   1.5.0            BSD License
 torch-cluster           1.5.7            MIT
 torch-geometric         1.5.0            MIT
 torch-scatter           2.0.5            MIT
 torch-sparse            0.6.7            MIT
 torch-spline-conv       1.2.0            MIT
 torchvision             0.6.0a0+82fd1c8  BSD
 tqdm                    4.56.0           MIT License, Mozilla Public License 2.0 (MPL 2.0)

```