import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

import numpy as np
import tqdm
from xfraud.glib.graph_loader import GraphData, NaiveHetGraph
from xfraud.glib.utils import timeit


def feature_mock(layer_data, graph, feature_dim=16):
    feature = {}
    times = {}
    indxs = {}
    texts = []
    for ntype, data in layer_data.items():
        if not data:
            continue
        idx = np.array(list(data.keys()))
        idx_data = np.array(list(data.values()))
        ts = idx_data[:, 1]

        feature[ntype] = np.zeros((len(idx), feature_dim))

        times[ntype] = ts
        indxs[ntype] = idx
    return feature, times, indxs, texts


def create_naive_het_graph_from_edges(df):
    logger = logging.getLogger('factory-naive-het-graph')
    logger.setLevel(logging.INFO)

    with timeit(logger, 'node-type-init'):
        view = df[['src', 'ts']].drop_duplicates()
        node_ts = dict((k, v) for k, v in view.itertuples(index=False))
        view = df[['src', 'src_type']].drop_duplicates()
        node_type = dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        )
        view = df[['dst', 'dst_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))

    if 'graph_edge_type' not in df:
        df['graph_edge_type'] = 'default'

    with timeit(logger, 'edge-list-init'):
        edge_list = list(
            df[['src', 'dst', 'graph_edge_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]

    select = df['seed'] > 0
    view = df[select][['src', 'src_label']].drop_duplicates()
    seed_label = dict((k, v) for k, v in view.itertuples(index=False))

    return NaiveHetGraph(node_type, edge_list,
                         seed_label=seed_label, node_ts=node_ts)


def create_graph_data_from_edges(df):
    node_link_ts = df[['src', 'ts']].drop_duplicates()
    node_ts = dict(
        (node, ts)
        for node, ts in node_link_ts.itertuples(index=False)
    )

    view = df[['src', 'src_type']].drop_duplicates()
    node_type = dict(
        (node, tp)
        for node, tp in view.itertuples(index=False)
    )
    view = df[['dst', 'dst_type']].drop_duplicates()
    node_type.update(dict(
        (node, tp)
        for node, tp in view.itertuples(index=False)
    ))

    view = df[['src', 'src_label']].drop_duplicates()
    node_label = dict(
        (node, lbl)
        for node, lbl in view.itertuples(index=False)
    )

    if 'graph_edge_type' not in df:
        df['graph_edge_type'] = 'default'

    type_adj = {}
    node_gtypes = defaultdict(set)
    graph_edge_type = {}
    for (stype, etype, dtype), gdf in df.groupby(
            ['src_type', 'graph_edge_type', 'dst_type']):
        gtype = stype, etype, dtype
        adj = defaultdict(set)
        for u, v in gdf[['src', 'dst']].itertuples(index=False):
            node_gtypes[u].add(gtype)
            node_gtypes[v].add(gtype)
            adj[u].add(v)
            adj[v].add(u)
        type_adj[gtype] = dict((k, tuple(v)) for k, v in adj.items())
        graph_edge_type[gtype] = etype

    rval = GraphData(
        type_adj=type_adj,
        node_gtypes=node_gtypes,
        node_ts=node_ts, node_type=node_type,
        graph_edge_type=graph_edge_type,
        node_label=node_label)
    return rval


def create_naive_het_homo_graph_from_edges(df):
    logger = logging.getLogger('factory-naive-het-homo-graph')
    logger.setLevel(logging.INFO)

    with timeit(logger, 'node-type-init'):
        view = df[['src', 'src_ts']].drop_duplicates()
        node_ts = dict((node, ts)
                       for node, ts in view.itertuples(index=False)
                       )
        view = df[['src']].drop_duplicates()
        node_type = dict(
            (node[0], 'node_link_id')
            for node in view.itertuples(index=False)
        )

    with timeit(logger, 'node-seed-init'):
        select = df['src_seed'] > 0
        view = df[select][['src', 'src_label']].drop_duplicates()
        seed_label = dict((k, v) for k, v in view.itertuples(index=False))

    with timeit(logger, 'edge-list-init'):
        # edge_list = []
        # df_tmp = df[['src', 'dst']].drop_duplicates()
        # for i, row in tqdm.tqdm(df_tmp.iterrows(),
        #                         total=df_tmp.shape[0],
        #                         desc='iter-edges'):
        #     edge_list.append(tuple(row.tolist()) + ('default',))

        view = df[['src', 'dst']].drop_duplicates()
        view['graph_edge_type'] = 'default'

        edge_list = view.to_numpy().tolist()

    return NaiveHetGraph(node_type, edge_list,
                         seed_label=seed_label, node_ts=node_ts)
