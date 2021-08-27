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
import pandas as pd
import numpy as np
import networkx as nx
import tqdm
import glob
import fire

from xfraud.glib.fstore import FeatureStore


def main(path_g='./data/g_publish.parquet',
         path_db='./data/feat_store_publish.db',
         path_feat='./data/feat_publish.parquet'):

    store = FeatureStore(path_db)

    df = pd.read_parquet(path_feat)
    columns = list(df.columns)
    columns.remove('node_link_id')
    df[columns] = df[columns].fillna(-1)
    df = df[['node_link_id']+columns]
    x = df.values
    with store.db.write_batch() as wb:
        for i in range(x.shape[0]):
            key = x[i, 0]
            value = x[i, 1:]
            store.put(key, value, wb=wb, dtype=np.float32)

    # add average feats of neighbour txn nodes to entity nodes
    df_graph = pd.read_parquet(path_g)
    node_src = df_graph['src'].drop_duplicates().tolist()
    node_dst = df_graph['dst'].drop_duplicates().tolist()
    graph = nx.from_pandas_edgelist(df_graph, source='src', target='dst')

    for node_id in tqdm.tqdm(node_dst):
        sum_feat = None
        cnt_neighbor = 0
        for neighbor in graph.neighbors(node_id):
            if neighbor in node_src:
                neighbor_feat = store.get(key=neighbor, default_value=None)
                if sum_feat is None:
                    sum_feat = neighbor_feat.copy()
                else:
                    sum_feat += neighbor_feat
                cnt_neighbor += 1
        store.put(key=node_id, value=sum_feat/cnt_neighbor)


if __name__=="__main__":
    fire.Fire(main)