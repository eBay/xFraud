import logging
logging.basicConfig(level=logging.INFO)

import fire
import numpy as np
import pandas as pd

from xfraud.glib.pyhgt_utils import load_graph
from xfraud.glib.brisk_utils import create_graph_data_loader_from_edges, feature_mock
from xfraud.glib.utils import timeit


def main(path_graph_edge, sample_depth=2, sample_number=16):
    logger = logging.getLogger('main')

    with timeit(logger, 'load-edge'):
        df = pd.read_parquet(path_graph_edge)
    with timeit(logger, 'load-graph-loader'):
        gdl = create_graph_data_loader_from_edges(df)
    with timeit(logger, 'load-pyhgt-graph'):
        graph = load_graph(df)

    times = pd.Series(list(graph.times))
    times_train_valid_split = times.quantile(0.8)
    times_valid_test_split = times.quantile(0.9)
    train_range = {t: True
                   for t in graph.times
                   if t != None and t <= times_train_valid_split}
    valid_range = {t: True
                   for t in graph.times
                   if
                   t != None and times_train_valid_split < t <= times_valid_test_split}
    test_range = {t: True
                  for t in graph.times
                  if t != None and t > times_valid_test_split}
    logger.info('Range Train %s\t Valid %s\t Test %s',
                train_range, valid_range, test_range)
    
    dvc = df['dst'].value_counts()
    dvc = dvc[dvc>1]
    src = df[df['dst'].isin(dvc.index)]['src'].unique()
    gfl = graph.node_forward['node_link_id']
    target_ids = {'node_link_id': [(gfl[e], gdl.node_ts[e]) for e in src]}

    with timeit(logger, 'subsample'):
        from pyHGT.data import sample_subgraph
        sample_subgraph(
            graph, time_range=train_range, inp=target_ids,
            sampled_depth=sample_depth,
            sampled_number=sample_number, feature_extractor=feature_mock)


if __name__ == '__main__':
    fire.Fire(main)
