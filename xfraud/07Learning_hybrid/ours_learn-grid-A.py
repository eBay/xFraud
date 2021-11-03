#!/usr/bin/env python
# coding: utf-8

# - Edge weight is inferred by GNNExplainer and node importance is given by five Ebay annotators. Not every annotator has annotated each node.
# - Seed is the txn to explain.
# - id is the community id. 

import math
from tqdm.auto import tqdm
import random
import pandas as pd
import numpy as np
import networkx as nx
import itertools
from collections import Counter
import scipy.stats
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--random-draw', action = 'store', dest = 'random_draw', type = int, default = 100, help = 'Random draws to break the tie in ranking topk edges.')
parser.add_option('--edge-agg', action = 'store', dest = 'edge_agg', default = 'avg', choices = ['avg', 'min', 'sum'], help = 'Aggregation method to compute edge importance score based on the node importance scores.')
# parser.add_option('--explainer-w', action = 'store', dest = 'explainer_w', default = '0', type = float, help = 'Learned parameter for explainer weights.')
parser.add_option('-c', '--centrality-w', action = 'store', dest = 'centrality_w', default = '0', type = float, help = 'Learned parameter for centrality measures.')

(options, args) = parser.parse_args()
print ("Options:", options)

explainer_w = 1-options.centrality_w
learner = 'grid-{}'.format(options.centrality_w)


# Load in the annotation file, the data seed, the edge weights by explainer, the edges in the communities.
DataNodeImp = pd.read_csv('../05GNNExplainer-eval-hitrate/input/annotation_publish.csv')
DataSeed = pd.read_csv('../05GNNExplainer-eval-hitrate/input/data-seed.txt')
DataEdgeWeight = pd.read_csv('../05GNNExplainer-eval-hitrate/input/data-edge-weight.txt')
df_e = pd.read_csv('../05GNNExplainer-eval-hitrate/input/masked_df_e.csv')

x_y_df = pd.read_csv('x_y_df_learn.csv')


del x_y_df['max_hitrate']


x_y_df['combined_weights'] = explainer_w * x_y_df['explainer'] + options.centrality_w * x_y_df['edge_btw']


print('AUC score of the sample:', roc_auc_score(DataSeed.y, DataSeed['yhat']))

# Communities labeled 0 and 1.
comm0 = DataSeed[DataSeed.y==0].id.unique()
comm1 = DataSeed[DataSeed.y==1].id.unique()


df_node_weight = pd.read_csv('./results/df_node_weight_with_avgimp.csv')



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

    if options.edge_agg == 'min':
        edge_imp_annotate = min(src_row['importance_avg'], dst_row['importance_avg']) 
    if options.edge_agg == 'avg':
        edge_imp_annotate = np.mean([src_row['importance_avg'], dst_row['importance_avg']])
    if options.edge_agg == 'sum':
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



df_edge_weight['combined_weights'] = x_y_df['combined_weights']



# Avg edge/community.
print('Average edges per community:', df_edge_weight.shape[0]/41)

df_edge_weight.rename(columns={'source':'src', 'target': 'dst', 'community_id': 'id', 
                       'importance': 'edge_importance', 'combined_weights': 'edge_weight'}, inplace=True)
df_edge_weight.to_csv('./results/df_edge_weight_imp-{}-{}.csv'.format(options.edge_agg, learner))
df_edge_weight.rename(columns={'edge_importance':'importance'}, inplace=True)



df_edge_weight = df_edge_weight.reset_index()



# Topk hit rate
hitrate_df = pd.DataFrame(index=['all', 'comm0', 'comm1'] + list(range(0,41)))

for k in [i*5 for i in range(1,11)]:
    hitrate_list_topk_comm = []    
    for cid in df_edge_weight.id.unique():
        df_edge_weight_sub = df_edge_weight[df_edge_weight.id==cid]
        
        imp_largest = sorted(dict(Counter(df_edge_weight_sub.importance)).items(), reverse=True)[0][0]
        count_largest = sorted(dict(Counter(df_edge_weight_sub.importance)).items(), reverse=True)[0][1]
        
        hitrate_list_topk = []
        
        for r in tqdm(range(0,options.random_draw), total=options.random_draw, ncols=80, mininterval=5):            
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

hitrate_df.to_csv('./results/topk-{}-{}-{}.csv'.format(options.random_draw, options.edge_agg, learner), index=True)
	
ours = hitrate_df.loc[['all', 'comm0', 'comm1']]
print('Our topk hit rate:', ours)

print(ours)


# In[17]:


train = hitrate_df.loc[range(0,21)]
test = hitrate_df.loc[range(21, 41)]
all = hitrate_df.loc[range(0, 41)]
np.mean(train).to_csv('./results/ours_{}_train.csv'.format(learner))
np.mean(test).to_csv('./results/ours_{}_test.csv'.format(learner))
np.mean(all).to_csv('./results/ours_{}_all.csv'.format(learner))

