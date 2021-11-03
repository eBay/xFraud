#!/usr/bin/env python
# coding: utf-8

# - Edge weight is inferred by GNNExplainer and node importance is given by five Ebay annotators. Not every annotator has annotated each node.
# - Seed is the txn to explain.
# - id is the community id. 

import os
import pickle
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
from scipy.stats import ks_2samp
import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--random-draw', action = 'store', dest = 'random_draw', type = int, default = 100, help = 'Random draws to break the tie in ranking topk edges.')
parser.add_option('--edge-agg', action = 'store', dest = 'edge_agg', default = 'avg', choices = ['avg', 'min', 'sum'], help = 'Aggregation method to compute edge importance score based on the node importance scores.')
parser.add_option('--type-centrality', action = 'store', dest = 'type_centrality', default = 'degree', choices = ['degree', 'eigenvector_centrality', 'closeness', 'current_flow_closeness_centrality', 'betweenness_centrality', 'current_flow_betweenness_centrality', 'approximate_current_flow_betweenness_centrality', 'communicability_betweenness_centrality', 'load_centrality', 'subgraph_centrality', 'subgraph_centrality_exp', 'harmonic_centrality'], help = 'Edge centrality calculation method.')

(options, args) = parser.parse_args()
print ("Options:", options)



# Load in the annotation file, the data seed, the edge weights by explainer, the edges in the communities.
DataNodeImp = pd.read_csv('../05GNNExplainer-eval-hitrate/input/annotation_publish.csv')
DataSeed = pd.read_csv('../05GNNExplainer-eval-hitrate/input/data-seed.txt')
DataEdgeWeight = pd.read_csv('../05GNNExplainer-eval-hitrate/input/data-edge-weight.txt')
df_e = pd.read_csv('../05GNNExplainer-eval-hitrate/input/masked_df_e.csv')


# Communities labeled 0 and 1.
comm0 = DataSeed[DataSeed.y==0].id.unique()
comm1 = DataSeed[DataSeed.y==1].id.unique()


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
df_node_weight.to_csv('./results/df_node_weight_with_avgimp.csv')


edge_agg = 'avg'


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

    if edge_agg == 'min':
        edge_imp_annotate = min(src_row['importance_avg'], dst_row['importance_avg']) 
    if edge_agg == 'avg':
        edge_imp_annotate = np.mean([src_row['importance_avg'], dst_row['importance_avg']])
    if edge_agg == 'sum':
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



# For each community, obtain the edge betweenness
df_edge_weight['edge_btw'] = 0

for community_id in df_edge_weight.community_id.unique().tolist():
    sub_df_edge_weight = df_edge_weight[df_edge_weight.community_id == community_id]
    G = nx.from_pandas_edgelist(sub_df_edge_weight, 'source', 'target')
    L = nx.line_graph(G)
    
    if options.type_centrality == 'degree': 
        centrality = nx.degree_centrality(L)
    if options.type_centrality == 'eigenvector_centrality':
        centrality = nx.eigenvector_centrality(L, max_iter=5000)
    if options.type_centrality == 'katz_centrality': # todo, find out max_iter
        centrality = nx.katz_centrality(L, max_iter=20000)
    if options.type_centrality == 'closeness':
        centrality = nx.closeness_centrality(L)
    if options.type_centrality == 'current_flow_closeness_centrality': # also known as information_centrality 
        try:
            centrality = nx.current_flow_closeness_centrality(L)
        except Exception as e: # NetworkXError: Graph not connected.
            print(e)
            print('Setting centrality to 0 in badly connected graphs.')
            centrality = {}
            for n in L.nodes:
                centrality[n] = 0  
    if options.type_centrality == 'betweenness_centrality':
        centrality = nx.betweenness_centrality(L, normalized = True)
    if options.type_centrality == 'current_flow_betweenness_centrality':
        try:
            centrality = nx.current_flow_betweenness_centrality(L)
        except Exception as e: # NetworkXError: Graph not connected.
            print(e)
            print('Setting centrality to 0 in badly connected graphs.')
            centrality = {}
            for n in L.nodes:
                centrality[n] = 0  
    if options.type_centrality == 'approximate_current_flow_betweenness_centrality':
        try:
            centrality = nx.approximate_current_flow_betweenness_centrality(L)
        except Exception as e: # NetworkXError: Graph not connected.
            print(e)
            print('Setting centrality to 0 in badly connected graphs.')
            centrality = {}
            for n in L.nodes:
                centrality[n] = 0          
    if options.type_centrality == 'communicability_betweenness_centrality':
        centrality = nx.communicability_betweenness_centrality(L)
    if options.type_centrality == 'load_centrality':
        centrality = nx.load_centrality(L)
    if options.type_centrality == 'subgraph_centrality':
        centrality = nx.subgraph_centrality(L)
    if options.type_centrality == 'subgraph_centrality_exp':
        centrality = nx.subgraph_centrality_exp(L)
    if options.type_centrality == 'harmonic_centrality':
        centrality = nx.harmonic_centrality(L)
    # Add betweenness to the dataframe df_edge_weight
    for ((src, tgt), score) in centrality.items():
        idx = df_edge_weight.index[(df_edge_weight.source == src)                                   &(df_edge_weight.target == tgt)                                   &(df_edge_weight.community_id == community_id)]
        df_edge_weight.loc[idx, 'edge_btw'] = score


df_edge_weight['weight'] = df_edge_weight['weight'].astype(float)


df_edge_weight['weight'].corr(df_edge_weight['edge_btw'])


# Attach betweenness to the dataframe

# Avg edge/community.
print('Average edges per community:', df_edge_weight.shape[0]/41)

# Use edge_btw as edge_weight
df_edge_weight.rename(columns={'source':'src', 'target': 'dst', 'community_id': 'id', 
                       'importance': 'edge_importance', 'edge_btw': 'edge_weight'}, inplace=True)

df_edge_weight.to_csv('./results/df_edge_weight_imp-{}-line-graph-{}.csv'.format(edge_agg, options.type_centrality)) # todo in main script
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
        
        for r in tqdm(range(0,100), total=100, ncols=80, mininterval=5): # todo in main script            
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

hitrate_df.to_csv('./results/topk-{}-{}-line-graph-{}.csv'.format(100, edge_agg, options.type_centrality), index=True) # todo in main script   

ours = hitrate_df.loc[['all', 'comm0', 'comm1']]
print('Human annotation vs. edge betweenness (topk hit rate):', ours)


if os.path.exists('./results/results.pkl'):
    results_pickle = pickle.load(open('./results/results.pkl', 'rb'))
else:
    results_pickle = {}
results_pickle['explainer'] = [0.44521951219512196,0.6919512195121952,0.8207479674796748,0.8984024390243902,0.9205951219512195]


#store list of results in a pickled file
results_pickle[options.type_centrality] = ours.iloc[0].values


pickle.dump(results_pickle, open('./results/results.pkl', 'wb'))


# show the results_pickle in a pandas df

results_df = pd.DataFrame.from_dict(results_pickle, orient='index',
                       columns=['top5', 'top10', 'top15', 'top20', 'top25'])



print(results_df)



if os.path.exists('./results/corr.pkl'):
    corr_pickle = pickle.load(open('./results/corr.pkl', 'rb'))
else:
    corr_pickle = {}
    
corr_pickle[options.type_centrality] = df_edge_weight['weight'].corr(df_edge_weight['edge_weight'])

pickle.dump(corr_pickle, open('./results/corr.pkl', 'wb'))
print(pd.DataFrame.from_dict(corr_pickle, orient='index',
                       columns=['./results/corr']))


if os.path.exists('./results/KS_test_stats.pkl'):
    KS_test_stats_pickle = pickle.load(open('./results/KS_test_stats.pkl', 'rb'))
else:
    KS_test_stats_pickle = {}

KS_test_stats_pickle[options.type_centrality] = [ks_2samp(df_edge_weight.weight, df_edge_weight.edge_weight).statistic,                                         ks_2samp(df_edge_weight.weight, df_edge_weight.edge_weight).pvalue,                                         ks_2samp(df_edge_weight.weight, df_edge_weight.edge_weight*2).statistic,                                         ks_2samp(df_edge_weight.weight, df_edge_weight.edge_weight*2).pvalue]

pickle.dump(KS_test_stats_pickle, open('./results/KS_test_stats.pkl', 'wb'))
print(pd.DataFrame.from_dict(KS_test_stats_pickle, orient='index',
                       columns=['stats', 'pvalue', 'x2_stats', 'x2_pvalue']))


df_edge_weight.to_csv('./results/edge_betweenness_inspection_{}.csv'.format(options.type_centrality))



