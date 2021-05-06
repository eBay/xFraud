import logging

import tqdm


def load_graph(df_edges):
    logger = logging.getLogger('load-graph')
    logger.setLevel(logging.INFO)

    try:
        from pyHGT.data import Graph
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
