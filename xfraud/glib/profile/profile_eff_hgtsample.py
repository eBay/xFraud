import logging

logging.basicConfig(level=logging.INFO)

import fire
import pandas as pd

from xfraud.glib.brisk_utils import create_graph_data_from_edges, feature_mock
from xfraud.glib.utils import timeit


def main(path_graph_edge, sample_depth=2, sample_number=16):
    logger = logging.getLogger('main')

    with timeit(logger, 'load-edge'):
        df = pd.read_parquet(path_graph_edge)
    with timeit(logger, 'load-graph-loader'):
        gdl = create_graph_data_from_edges(df)

    times = pd.Series(list(df['ts'].unique()))
    times_train_valid_split = times.quantile(0.8)
    times_valid_test_split = times.quantile(0.9)
    train_range = {t: True
                   for t in times
                   if t != None and t <= times_train_valid_split}
    valid_range = {t: True
                   for t in times
                   if
                   t != None and times_train_valid_split < t <= times_valid_test_split}
    test_range = {t: True
                  for t in times
                  if t != None and t > times_valid_test_split}
    logger.info('Range Train %s\t Valid %s\t Test %s',
                train_range, valid_range, test_range)

    dvc = df['dst'].value_counts()
    dvc = dvc[dvc > 1]
    seeds = df[df['dst'].isin(dvc.index)]['src'].unique()

    with timeit(logger, 'subsample'):
        gdl.sample(seeds=seeds,
                   depth=sample_depth, width=sample_number,
                   ts_max=max(train_range))


if __name__ == '__main__':
    fire.Fire(main)
