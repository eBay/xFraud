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


import logging
import tqdm


def load_graph(df_edges):
    logger = logging.getLogger('load-graph')
    logger.setLevel(logging.INFO)

    try:
        from third_party.pyHGT.pyHGT.data import Graph
    except ImportError as e:
        logger.critical(
            'Please make sure that original pyHGT package is '
            'available in PYTHONPATH!')
        raise e

    g = Graph()
    logger.info('#node %d, #edge %d.',
                len(set(df_edges['src']).union(set(df_edges['dst']))),
                len(df_edges))

    for src, dst, ts, slabel, stype, dtype in tqdm.tqdm(
            df_edges[['src', 'dst', 'ts', 'src_label', 'src_type',
                      'dst_type']].itertuples(index=False),
            desc='load-graph',
            total=len(df_edges)):
        g.add_edge({'id': src, 'type': stype},
                   {'id': dst, 'type': dtype}, time=ts, directed=False)
        if stype not in g.node_feature:
            g.node_feature[stype] = []
        if dtype not in g.node_feature:
            g.node_feature[dtype] = []
    return g
