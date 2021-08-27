# Copyright 2020-2021 eBay Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# - Edge weight is inferred by GNNExplainer and node importance is given by five Ebay annotators. Not every annotator has annotated each node.
# - Seed is the txn to explain.
# - id is the community id. 

from tqdm.auto import tqdm
import random
import pandas as pd
import numpy as np
import itertools
from collections import Counter
import sklearn
from sklearn.metrics import roc_auc_score
import warnings
from argparse import ArgumentParser

warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--random-draw', action='store', dest='random_draw', type=int, default=100, help='Random draws to break the tie in ranking topk edges.')
parser.add_argument('--edge-agg', action='store', dest='edge_agg', default='avg', choices=['avg', 'min', 'sum'], help='Aggregation method to compute edge importance score based on the node importance scores.')

args = parser.parse_args()
print("Options:", args)

# Load in the annotation file, the data seed, the edge weights by explainer, the edges in the communities.
DataNodeImp = pd.read_csv('./xfraud/explainer-eval-hitrate/input/annotation_publish.csv')
DataSeed = pd.read_csv('./xfraud/explainer-eval-hitrate/input/data-seed.txt')
DataEdgeWeight = pd.read_csv('./xfraud/explainer-eval-hitrate/input/data-edge-weight.txt')
df_e = pd.read_csv('./xfraud/explainer-eval-hitrate/input/masked_df_e.csv')

print('AUC score of the sample:', roc_auc_score(DataSeed.y, DataSeed['yhat']))

# Communities labeled 0 and 1.
comm0 = DataSeed[DataSeed.y==0].id.unique()
comm1 = DataSeed[DataSeed.y==1].id.unique()

# check that every edge has at least two annotators.
less_than_two_rater_idx = []
for i, row in DataNodeImp.iterrows():
    na_count = sum([pd.isna(DataNodeImp.loc[i, 'importance_annotator{}'.format(user)]) for user in range(1,6)])
    if na_count >=4:
        less_than_two_rater_idx.append(i)
print('nodes annotated by less than two annotators:', len(less_than_two_rater_idx))

# IAA score.
IAA_list_human = []
for pair in itertools.combinations(range(1,6), 2):
    IAA_df = DataNodeImp[(~DataNodeImp['importance_annotator{}'.format(pair[0])].isnull()) & (~DataNodeImp['importance_annotator{}'.format(pair[1])].isnull())]
    IAA = sklearn.metrics.cohen_kappa_score(IAA_df['importance_annotator{}'.format(pair[0])], IAA_df['importance_annotator{}'.format(pair[1])])
    IAA_list_human.append(IAA)
    print('human pair {} IAA: {}'.format(pair, IAA))

# Random annotation vs. human annotation: 
# For every annotation given by an annotator, swap it with a random integer among 0, 1, 2 (2 inclusive)
print('Random annotator at work ... ')
IAA_mean_rand_list = []
for _ in range(0,10):
    random.seed(_)
    DataNodeImp_rand = DataNodeImp.copy()
    for i, row in tqdm(DataNodeImp_rand.iterrows(), total=len(DataNodeImp_rand), ncols=80, mininterval=5):
        for user in range(1,6):
            anno = DataNodeImp_rand.loc[i, 'importance_annotator{}'.format(user)]
            if not pd.isna(anno):
                rand_anno = random.randrange(3)
                DataNodeImp_rand.loc[i, 'importance_annotator{}'.format(user)] = rand_anno
    
    IAA_list = []
    for pair in itertools.combinations(range(1,6), 2):
        IAA_df = DataNodeImp_rand[(~DataNodeImp_rand['importance_annotator{}'.format(pair[0])].isnull()) & (~DataNodeImp_rand['importance_annotator{}'.format(pair[1])].isnull())]
        IAA = sklearn.metrics.cohen_kappa_score(IAA_df['importance_annotator{}'.format(pair[0])], IAA_df['importance_annotator{}'.format(pair[1])])
        IAA_list.append(IAA)
        # print('random pair {} IAA: {}'.format(pair, IAA))
        
    IAA_mean = np.mean(pd.Series(IAA_list).dropna())
    IAA_mean_rand_list.append(IAA_mean)

print('IAA score among the annotators vs. random:', np.mean(IAA_list_human), np.mean(IAA_mean_rand_list))

# Preprocess annotation score: calculate average importance for node.
DataNodeImp['importance_avg'] = None
for i, row in DataNodeImp.iterrows():
    sum_importance = 0
    cnt_user = 0
    for user in range(1,6):
        if not pd.isna(row['importance_annotator{}'.format(user)]):
            sum_importance += row['importance_annotator{}'.format(user)]
            cnt_user += 1
    if cnt_user == 0:
        DataNodeImp.loc[i, 'importance_avg'] = 0
    else:
        DataNodeImp.loc[i, 'importance_avg'] = sum_importance/cnt_user
df_node_weight = DataNodeImp.copy()   
print('Value counts of node importance scores:', Counter(df_node_weight.importance_avg))
# df_node_weight.to_csv('./results/df_node_weight_with_avgimp.csv')

# Preprocess explainer weights: calculate undirectional edge weight by taking the max weight of bidirectional edge weights.
# From node importance scores to edge importance score: "min"/"avg"/"sum". 
df_edge_weight = df_e.copy()
df_edge_weight['importance'] = None
df_edge_weight['weight'] = None
df_edge_weight['weight_positive'] = None
df_edge_weight['weight_negative'] = None

for i, row in tqdm(df_edge_weight.iterrows(), total=len(df_edge_weight), ncols=80, mininterval=5):
    src_node_id = row['source']
    dst_node_id = row['target']
    cc_id = row['community_id']
  
    src_row = df_node_weight[(df_node_weight['node_id']==src_node_id) & (df_node_weight['community_id']==cc_id)].iloc[0]
    dst_row = df_node_weight[(df_node_weight['node_id']==dst_node_id) & (df_node_weight['community_id']==cc_id)].iloc[0]

    if args.edge_agg == 'min':
        edge_imp_annotate = min(src_row['importance_avg'], dst_row['importance_avg']) 
    if args.edge_agg == 'avg':
        edge_imp_annotate = np.mean([src_row['importance_avg'], dst_row['importance_avg']])
    if args.edge_agg == 'sum':
        edge_imp_annotate = src_row['importance_avg'] + dst_row['importance_avg']

    edge_weights = DataEdgeWeight[DataEdgeWeight['src'].isin([src_node_id, dst_node_id]) & 
                                  DataEdgeWeight['dst'].isin([src_node_id, dst_node_id]) & 
                                  DataEdgeWeight['id'].isin([cc_id])]['edge_weight'].max()
    
    df_edge_weight['importance'].iloc[i] = edge_imp_annotate
    df_edge_weight['weight'].iloc[i] = edge_weights
    df_edge_weight['weight_positive'].iloc[i] = DataEdgeWeight[DataEdgeWeight['src'].isin([src_node_id]) & 
                                                               DataEdgeWeight['dst'].isin([dst_node_id]) & 
                                                               DataEdgeWeight['id'].isin([cc_id])]['edge_weight'].iloc[0]
    df_edge_weight['weight_negative'].iloc[i] = DataEdgeWeight[DataEdgeWeight['src'].isin([dst_node_id]) & 
                                                               DataEdgeWeight['dst'].isin([src_node_id]) & 
                                                               DataEdgeWeight['id'].isin([cc_id])]['edge_weight'].iloc[0]

# Avg edge/community.
print('Average edges per community:', df_edge_weight.shape[0]/41)

df_edge_weight.rename(columns={'source':'src', 'target': 'dst', 'community_id': 'id', 
                       'importance': 'edge_importance', 'weight': 'edge_weight'}, inplace=True)
# df_edge_weight.to_csv('./results/df_edge_weight_imp-{}.csv'.format(args.edge_agg))
df_edge_weight.rename(columns={'edge_importance':'importance'}, inplace=True)

# Topk hit rate
hitrate_df = pd.DataFrame(index=['all', 'comm0', 'comm1'] + list(range(0,41)))

for k in [i*5 for i in range(1,6)]:
    hitrate_list_topk_comm = []    
    for cid in df_edge_weight.id.unique():
        df_edge_weight_sub = df_edge_weight[df_edge_weight.id==cid]
        
        imp_largest = sorted(dict(Counter(df_edge_weight_sub.importance)).items(), reverse=True)[0][0]
        count_largest = sorted(dict(Counter(df_edge_weight_sub.importance)).items(), reverse=True)[0][1]
        
        hitrate_list_topk = []
        
        for r in tqdm(range(0, args.random_draw), total=args.random_draw, ncols=80, mininterval=5):
            random.seed(r)
            if count_largest <= k:
                src_id_human_topk = df_edge_weight_sub[['src','dst']].values.tolist()
            else:
                all_human_top_edge_idx = df_edge_weight_sub[df_edge_weight_sub.importance == imp_largest].index
                human_topk_edge_idx = random.sample(list(all_human_top_edge_idx), k)
                src_id_human_topk = df_edge_weight.iloc[human_topk_edge_idx][['src','dst']].values.tolist()

            explainer_topk_edge = df_edge_weight_sub.sort_values(by=['edge_weight'], ascending=False)[['edge_weight', 'src', 'dst']][:k]
            src_id_explainer_topk = explainer_topk_edge[['src','dst']].values.tolist()

            hitrate = len([p for p in src_id_explainer_topk if p in src_id_human_topk or (p[1], p[0]) in src_id_human_topk])/k
            hitrate_list_topk.append(hitrate)
        hitrate_list_topk_comm.append(np.mean(hitrate_list_topk))
            
    all_hitrate = np.mean(hitrate_list_topk_comm)
    comm0_hitrate = np.mean([h for (i,h) in enumerate(hitrate_list_topk_comm) if i in comm0])
    comm1_hitrate = np.mean([h for (i,h) in enumerate(hitrate_list_topk_comm) if i in comm1])  
    
    hitrate_df['top{}'.format(k)] = [all_hitrate, comm0_hitrate, comm1_hitrate] + hitrate_list_topk_comm

# hitrate_df.to_csv('./results/topk-{}-{}.csv'.format(args.random_draw, args.edge_agg), index=True)

ours = hitrate_df.loc[['all', 'comm0', 'comm1']]
print('Our topk hit rate:', ours)







