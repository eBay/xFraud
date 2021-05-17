# xFraud

This is an open source version of xFraud code https://arxiv.org/abs/2011.12193

xFraud is an explainable Fraud transaction prediction system. xFraud is composed of a predictor which learns expressive 
representations for malicious transaction detection from the heterogeneous transaction graph via a self-attentive 
heterogeneous graph neural network, and an explainer that generates meaningful and human understandable explanations 
from graphs to facilitate further process in business unit.

  - [Workflow](#workflow)
      - [1. Env setup](#1-env-setup)
      - [2. Feature store setup](#2-feature-store-setup)
      - [3. Detector: Training and testing](#3-detector-training-and-testing)
      - [4. Explainer](#4-explainer)
      - [5. Explainer quantitative analysis (top k hit rate)](#5-explainer-quantitative-analysis-top-k-hit-rate)
  - [Data files](#data-files)
  - [Note](#note)
  - [Reference](#reference)
## Workflow

#### 1. Env setup

Setup the python environment with `conda` and install `pytorch` and its dependencies. 

```bash
bash install-env-publish.sh
```

#### 2. Feature store setup

LevelDB is used to store node features. <br>
For this data sample, to generate the data store is not a necessary step, as we also provide the feature store 
(`./data/feat_store_publish.db`) used in the subsequent scripts.

```bash
python xfraud/setup_feature_store_publish.py
```

#### 3. Detector: Training and testing 

```bash
bash train_script_publish.sh
```
#### 4. Explainer

```bash 
python xfraud/run_explainer.py
```

#### 5. Explainer quantitative analysis (top k hit rate)

```bash
python xfraud/ours.py
python random-baseline.py
```

## Data files
We provide a small sample of the transaction graph and features in `./data`. <br>
We also provide the sample, its annotations, and evaluation results (in `./explainer-eval-hitrate`) we describe in 
Section 4.2. <br>
All the datafiles are described in the scripts that utilize them. 

## Note
We provide a small sample of dummy data (anonymized transaction records), the scripts of xFraud, as well as the 
evaluation sample and scripts as supplemental material. The data we use in the paper is proprietary, i.e., real-world 
transaction records on the eBay platform. At this point we can only share a small dummy dataset with the reviewers to 
show the data structure and our workflow. In the long run, it would be possible to share the eBay-small dataset after 
the legal review at eBay. Regarding the scripts, since we only share them for reviews, they should be protected. After 
we are through the legal review process at eBay, we will also open source them. We strongly believe that the 
availability and reproducibility of data and scripts can benefit both industries and academia. 

## Reference
- `./pyHGT` is from the original HGT implementation, 
see [https://github.com/acbull/pyHGT/tree/master/pyHGT](https://github.com/acbull/pyHGT/tree/master/pyHGT). 

## License Information
Copyright 2021 eBay Inc. <br>
Author/Developer: Wei Min, Zhichao Han, Zitao Zhang

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the 
License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0. <br>
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an 
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the 
specific language governing permissions and limitations under the License.

