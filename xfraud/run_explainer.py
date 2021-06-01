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


import os
import glob
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)


import fire
import tqdm
import joblib
import numpy as np
import pandas as pd

# torch
import torch

from torch_geometric.nn.models import GNNExplainer

from xfraud.glib.fstore import FeatureStore
from xfraud.glib.brisk_utils import create_naive_het_graph_from_edges as create_naive_het_graph_from_edges
from xfraud.glib.pyg.model import GNN, HetNet as Net
from xfraud.glib.utils import timeit

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def apply_edge(df, v, map_node):
    e = df
    
    df['src'] = df['source']
    df['dst'] = df['target']
    df['ts'] = -1
    df['src_label'] = -1
    df['seed'] = 0
    df['seed'] = -1
    df.loc[e['source']==v[v.propertiesSeed==1]['id'].iloc[0], 'seed'] = 1
    df['src_type'] = 'node_link_id'
    df['dst_type'] = df['label'].apply(lambda x: map_node[x.split('-')[-1]])
    df['seed'] = (df['propertiesSeed'] == '1').fillna(0).astype(int)
    return df[['src', 'dst', 'ts', 'src_label', 'src_type', 'dst_type', 'seed']]


def main(
    path_dump,
    path_feat_db,
    path_model, path_g,
    path_map_node,
    path_case,
    conv_name='het-emb',
    n_hid=400, n_heads=8, n_layers=6, dropout=0.2):

    """
    :param path_dump: path to dump output csv
    :param path_feat_db: path to feature store
    :param path_model: path to model parameters
    :param path_g: path to graph edge files
    :param path_map_node: path to node type mapping
    :param path_case: path to case files
    """

    store = FeatureStore(path_feat_db)

    with timeit(logger, 'edge-load'):
        df_edges = pd.read_parquet(path_g)

    x0 = store.get(df_edges.iloc[0]['src'], None)
    assert x0 is not None
    num_feat = x0.shape[0]

    map_node = joblib.load(path_map_node)

    def load_g_x(id_=0, root=path_case):
        path_edge = f'{root}/edgelist_raw{id_}.parquet'
        path_node = f'{root}/nodelist_raw{id_}.parquet'
        e = pd.read_parquet(path_edge)
        v = pd.read_parquet(path_node)
        e = apply_edge(e, v, map_node)
        g = create_naive_het_graph_from_edges(e)
        return v, e, g

    v, _, g = load_g_x(id_=0)

    num_node_type=len(g.node_type_encode)
    num_edge_type=len(g.edge_type_encode)
    logger.info('#node_type %d, #edge_type %d', num_node_type, num_edge_type)

    gnn = GNN(conv_name=conv_name, 
                n_in=num_feat, 
                n_hid =n_hid, n_heads=n_heads, n_layers=n_layers, 
                dropout=dropout, 
                num_node_type=num_node_type, 
                num_edge_type=num_edge_type
                )

    model = Net(gnn, num_feat, num_embed=n_hid, n_hidden=n_hid)
    model.load_state_dict(torch.load(path_model))
    model.to(device)

    docs = []
    for i, _ in enumerate(tqdm.tqdm(glob.glob(f'{path_case}/nodelist_raw*.parquet'))):
        v, e, g = load_g_x(id_=i)
        seed = v[v.propertiesSeed==1].id.tolist()[0]

        node_ids = [g.node_decode[i] for i in range(len(g.node_decode))]

        encode_new = dict((v, i) for i, v in enumerate(node_ids))
        x_default = np.zeros((num_feat,), np.float32)
        x = np.asarray(
            [store.get(nid, x_default) for nid in node_ids]
        )

        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(g.edge_list_encoded)
        node_type = torch.LongTensor([g.node_type_encode[g.node_type[e]] for e in node_ids])
        edge_type = torch.LongTensor(g.edge_list_type_encoded)
        node_idx = encode_new[seed]

        explainer = GNNExplainer(model, epochs=200)
        explainer.to(device)

        x = x.to(device)
        edge_index = edge_index.to(device)
        node_type = node_type.to(device)
        edge_type = edge_type.to(device)

        node_feat_mask, edge_mask = explainer.explain_node(
            node_idx, 
            x, 
            edge_index=edge_index, 
            node_type=node_type,
            edge_type=edge_type)

        edge_index_i = pd.DataFrame(edge_index.cpu().numpy().T, columns=['src', 'dst'])
        edge_index_i['id'] = i
        edge_index_i['edge_weight'] = edge_mask.cpu().numpy()
        docs.append(edge_index_i)
    pd.concat(docs).to_csv(path_dump, index=False)


if __name__ == '__main__':
    fire.Fire(main)
