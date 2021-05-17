#!/usr/bin/env python
# coding: utf-8

import math
import tqdm
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

(options, args) = parser.parse_args()
print ("Options:", options)

# Load in the annotation file with computed avg node importance, the data seed, the edge weights by explainer, the edges in the communities.
DataNodeImp = pd.read_csv('./results/df_node_weight_with_avgimp.csv')
DataSeed = pd.read_csv('./input/data-seed.txt')
DataEdgeWeight = pd.read_csv('./input/data-edge-weight.txt')
df_e = pd.read_csv('./input/masked_df_e.csv')
df_node_weight = DataNodeImp.copy()

comm0 = DataSeed[DataSeed.y==0].id.unique()
comm1 = DataSeed[DataSeed.y==1].id.unique()

hitrate_df_randruns = pd.DataFrame(index=['all', 'comm0', 'comm1'] + list(range(0,41)))
RandDataEdgeWeight_df = pd.DataFrame()

print(f'Randomly assigning edge weights {options.random_draw} times ...')
for _ in range(0,10):
    
    random.seed(_)
    DataEdgeWeight = pd.read_csv('./input/data-edge-weight.txt')
    rand_weight = [random.random() for i in range(0, DataEdgeWeight.shape[0])]
    RandDataEdgeWeight = DataEdgeWeight
    RandDataEdgeWeight['edge_weight'] = rand_weight
    DataEdgeWeight = RandDataEdgeWeight
    RandDataEdgeWeight_df = pd.concat([RandDataEdgeWeight_df, DataEdgeWeight['edge_weight']])
    
    # Preprocess explainer weights: calculate undirectional edge weight by taking the max weight of bidirectional edge weights.
    df_edge_weight = df_e.copy()
    df_edge_weight['importance'] = None
    df_edge_weight['weight'] = None
    df_edge_weight['weight_positive'] = None
    df_edge_weight['weight_negative'] = None

    for i, row in tqdm.tqdm(df_edge_weight.iterrows(), total=len(df_edge_weight), ncols=80, mininterval=5):
        src_node_id = row['source']
        dst_node_id = row['target']
        cc_id = row['community_id']


        src_row = df_node_weight[(df_node_weight['node_id']==src_node_id) & (df_node_weight['community_id']==cc_id)].iloc[0]
        dst_row = df_node_weight[(df_node_weight['node_id']==dst_node_id) & (df_node_weight['community_id']==cc_id)].iloc[0]

        edge_imp_annotate = min(src_row['importance_avg'], dst_row['importance_avg'])

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
        
        
        
    # Preprocess annotation score: # From node importance scores to edge importance score: "min"/"avg"/"sum". 
    df_edge_weight = DataEdgeWeight.copy()
    df_edge_weight['importance'] = None

    for i, row in tqdm.tqdm(df_edge_weight.iterrows(), total=len(df_edge_weight), mininterval=10):
        src_node_id = row['src']
        dst_node_id = row['dst']
        cc_id = row['id']

        src_row = df_node_weight[(df_node_weight['node_id']==src_node_id) & (df_node_weight['community_id']==cc_id)].iloc[0]
        dst_row = df_node_weight[(df_node_weight['node_id']==dst_node_id) & (df_node_weight['community_id']==cc_id)].iloc[0]

        if options.edge_agg == 'min':
            edge_imp_annotate = min(src_row['importance_avg'], dst_row['importance_avg']) 
        if options.edge_agg == 'avg':
            edge_imp_annotate = np.mean([src_row['importance_avg'], dst_row['importance_avg']])
        if options.edge_agg == 'sum':
            edge_imp_annotate = src_row['importance_avg'] + dst_row['importance_avg']
            
        df_edge_weight['importance'].iloc[i] = edge_imp_annotate
        
    df_edge_weight.importance = df_edge_weight.importance.fillna(0)

    # Calculate random scores.
    for k in [i*5 for i in range(1,6)]:
        hitrate_list_topk_comm = []

        for cid in df_edge_weight.id.unique():
            df_edge_weight_sub = df_edge_weight[df_edge_weight.id==cid]
            imp_largest = sorted(dict(Counter(df_edge_weight_sub.importance)).items(), reverse=True)[0][0]
            count_largest = sorted(dict(Counter(df_edge_weight_sub.importance)).items(), reverse=True)[0][1]

            hitrate_list_topk = []

            for r in tqdm.tqdm(range(0,options.random_draw)):

                random.seed(r)
                if count_largest <= k:
                    src_id_human_topk = df_edge_weight_sub[['src','dst']].values.tolist()
                else:
                    all_human_top_edge_idx = df_edge_weight_sub[df_edge_weight_sub.importance == imp_largest].index
                    human_topk_edge_idx = random.sample(list(all_human_top_edge_idx), k)
                    src_id_human_topk = df_edge_weight.iloc[human_topk_edge_idx][['src','dst']].values.tolist()

                explainer_topk_edge = df_edge_weight_sub.sort_values(by=['edge_weight'], ascending=False)[['src', 'dst']][:k]
                src_id_explainer_topk = explainer_topk_edge[['src','dst']].values.tolist()

                hitrate = len([p for p in src_id_explainer_topk if p in src_id_human_topk or (p[1], p[0]) in src_id_human_topk])/k
                # Hitrate of each random draw from the top ranked edges.
                hitrate_list_topk.append(hitrate)
            # Take the average the topk hitrate and append that to the community hitrate.
            hitrate_list_topk_comm.append(np.mean(hitrate_list_topk))
        
        # Averaging all the hitrates in communities of different characteristics
        all_hitrate = np.mean(hitrate_list_topk_comm)
        comm0_hitrate = np.mean([h for (i,h) in enumerate(hitrate_list_topk_comm) if i in comm0])
        comm1_hitrate = np.mean([h for (i,h) in enumerate(hitrate_list_topk_comm) if i in comm1])    

        hitrate_df_randruns[f'top{k}-rand{_}'] = [all_hitrate, comm0_hitrate, comm1_hitrate] + hitrate_list_topk_comm
hitrate_df_randruns.to_csv(f'./results/topk-hitrate-randruns{options.random_draw}-{options.edge_agg}.csv')

# Compute \delta(Ours-Random)
hitrate_df_avg = pd.DataFrame(index=['all', 'comm0', 'comm1'])

for k in [i*5 for i in range(1,6)]:
    all_hitrate_avg = np.mean(pd.Series([hitrate_df_randruns[f'top{k}-rand{i}']['all'] for i in range(0,10) ]).dropna())

    comm0_hitrate_avg = np.mean(pd.Series([hitrate_df_randruns[f'top{k}-rand{i}']['comm0'] for i in range(0,10) ]).dropna())

    comm1_hitrate_avg = np.mean(pd.Series([hitrate_df_randruns[f'top{k}-rand{i}']['comm1'] for i in range(0,10) ]).dropna())

    # print(f'top{k} random hitrate:{all_hitrate_avg}, {comm0_hitrate_avg}, {comm1_hitrate_avg}')

    hitrate_df_avg[f'top{k}'] = [all_hitrate_avg, comm0_hitrate_avg, comm1_hitrate_avg]

hitrate_df = pd.read_csv(f'./results/topk-{options.random_draw}-{options.edge_agg}.csv')
hitrate_df.set_index('Unnamed: 0', inplace=True)

hitrate_df_ours = hitrate_df[:3]
print('Ours:', hitrate_df_ours)
print('Random:', hitrate_df_avg)
diff = hitrate_df_ours - hitrate_df_avg
print('Diff:', diff)
diff.to_csv(f'./results/topk-ours-vs-randruns-{options.random_draw}-{options.edge_agg}.csv')
