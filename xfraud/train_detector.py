import os
import gzip
import logging
import shutil
import tempfile
import subprocess
from functools import partial
from collections import defaultdict
import glob

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)

import fire
import tqdm
import joblib
import numpy as np
import pandas as pd
import datetime

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.utils import convert_tensor
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.contrib.metrics import AveragePrecision, ROC_AUC
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

from xfraud.glib.fstore import FeatureStore
from xfraud.glib.brisk_utils import create_naive_het_graph_from_edges as _create_naive_het_graph_from_edges
from xfraud.glib.graph_loader import NaiveHetDataLoader, NaiveHetGraph
from xfraud.glib.pyg.model import GNN, HetNet as Net, HetNetLogi as NetLogi
from xfraud.glib.utils import timeit

mem = joblib.Memory('./data/cache')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

create_naive_het_graph_from_edges = mem.cache(_create_naive_het_graph_from_edges)


def prepare_batch(batch, ts_range, fstore, default_feature,
                  g: NaiveHetGraph,
                  device=device, non_blocking=False):
    encoded_seeds, encoded_ids, edge_ids = batch
    encoded_seeds = set(encoded_seeds)
    encode_to_new = dict((e, i) for i, e in enumerate(encoded_ids))
    mask = np.asarray([e in encoded_seeds for e in encoded_ids])
    decoded_ids = [g.node_decode[e] for e in encoded_ids]

    x = np.asarray([
        fstore.get(e, default_feature) for e in decoded_ids
    ])
    x = convert_tensor(torch.FloatTensor(x), device=device, non_blocking=non_blocking)

    edge_list = [g.edge_list_encoded[:, idx] for idx in edge_ids]
    f = lambda x: encode_to_new[x]
    f = np.vectorize(f)
    edge_list = [f(e) for e in edge_list]
    edge_list = [
        convert_tensor(torch.LongTensor(e), device=device, non_blocking=non_blocking)
        for e in edge_list]

    y = np.asarray([
        -1 if e not in encoded_seeds else g.seed_label_encoded[e]
        for e in encoded_ids
    ])
    assert (y >= 0).sum() == len(encoded_seeds)

    y = torch.LongTensor(y)
    y = convert_tensor(y, device=device, non_blocking=non_blocking)
    mask = torch.BoolTensor(mask)
    mask = convert_tensor(mask, device=device, non_blocking=non_blocking)

    y = y[mask]

    node_type_encode = g.node_type_encode
    node_type = [node_type_encode[g.node_type[e]] for e in decoded_ids]
    node_type = torch.LongTensor(np.asarray(node_type))
    node_type = convert_tensor(
        node_type, device=device, non_blocking=non_blocking)

    edge_type = [[g.edge_list_type_encoded[eid] for eid in list_] for list_ in edge_ids]
    edge_type = [torch.LongTensor(np.asarray(e)) for e in edge_type]
    edge_type = [convert_tensor(e, device=device, non_blocking=non_blocking) for e in edge_type]

    return ((mask, x, edge_list, node_type, edge_type), y)


def main(path_g, path_feat_db='data/store.db', path_result='exp_result.csv',
         dir_model='./model',
         conv_name='gcn', sample_method='sage',
         batch_size=(64, 16),
         width=16, depth=6,
         n_hid=400, n_heads=8, n_layers=6, dropout=0.2,
         optimizer='adamw', clip=0.25,
         n_batch=32, max_epochs=10, patience=8,
         seed_epoch=False, num_workers=0,
         seed=2020, debug=False, continue_training=False):
    """
    :param path_g:          path of graph file
    :param path_feat_db:    path of feature store db
    :param path_result:     path of output result csv file
    :param dir_model:       path of model saving
    :param conv_name:       model convolution layer type, choices ['', 'logi', 'gcn', 'gat', 'hgt', 'het-emb']
    :param sample_method:
    :param batch_size:      positive/negative samples per batch
    :param width:           sample width
    :param depth:           sample depth
    :param n_hid:           num of hidden state
    :param n_heads:
    :param n_layers:        num of convolution layers
    :param dropout:
    :param optimizer:
    :param clip:
    :param n_batch:
    :param max_epochs:
    :param patience:
    :param seed_epoch:      True -> iter on all seeds; False -> sample seed according to batch_size
    :param num_workers:
    :param seed:            random seed
    :param debug:           debug mode
    :param continue_training:
    :return:
    """

    if conv_name == '' or conv_name == 'logi':
        width, depth = 1, 1

    stats = dict(
        batch_size=batch_size,
        width=width, depth=depth,
        n_hid=n_hid, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        conv_name=conv_name, optimizer=str(optimizer), clip=clip,
        max_epochs=max_epochs, patience=patience,
        seed=seed, path_g=path_g,
        sample_method=sample_method, path_feat_db=path_feat_db,
    )
    logger.info('Param %s', stats)

    with tempfile.TemporaryDirectory() as tmpdir:
        path_feat_db_temp = f'{tmpdir}/store.db'

        with timeit(logger, 'fstore-init'):
            subprocess.check_call(
                f'cp -r {path_feat_db} {path_feat_db_temp}',
                shell=True)

            store = FeatureStore(path_feat_db_temp)

        if not os.path.isdir(dir_model):
            os.makedirs(dir_model)
        with timeit(logger, 'edge-load'):
            df_edges = pd.read_parquet(path_g)
        if debug:
            logger.info('Main in debug mode.')
            df_edges = df_edges.iloc[:10000]
        if 'seed' not in df_edges:
            df_edges['seed'] = 1
        with timeit(logger, 'g-init'):
            g = create_naive_het_graph_from_edges(df_edges)

        seed_set = set(df_edges.query('seed>0')['src'])
        logger.info('#seed %d', len(seed_set))

        times = pd.Series(df_edges['ts'].unique())
        times_train_valid_split = times.quantile(0.7)
        times_valid_test_split = times.quantile(0.9)
        train_range = set(t for t in times
                          if t is not None and t <= times_train_valid_split)
        valid_range = set(t for t in times
                          if t is not None and times_train_valid_split < t <= times_valid_test_split)
        test_range = set(t for t in times
                         if t is not None and t > times_valid_test_split)
        logger.info('Range Train %s\t Valid %s\t Test %s',
                    train_range, valid_range, test_range)

        x0 = store.get(g.get_seed_nodes(train_range)[0], None)
        assert x0 is not None
        num_feat = x0.shape[0]

        np.random.seed(seed)
        torch.manual_seed(seed)

        dl_train = NaiveHetDataLoader(
            width=width, depth=depth,
            g=g, ts_range=train_range, method=sample_method,
            batch_size=batch_size, n_batch=n_batch,
            seed_epoch=seed_epoch, num_workers=num_workers, shuffle=True)

        dl_valid = NaiveHetDataLoader(
            width=width, depth=depth,
            g=g, ts_range=valid_range, method=sample_method,
            batch_size=batch_size, n_batch=n_batch,
            seed_epoch=True, num_workers=num_workers, shuffle=False,
            cache_result=True)

        dl_test = NaiveHetDataLoader(
            width=width, depth=depth,
            g=g, ts_range=test_range, method=sample_method,
            batch_size=batch_size, n_batch=n_batch,
            seed_epoch=True, num_workers=num_workers, shuffle=False,
            cache_result=True)

        logger.info('Len dl train %d, valid %d, test %d.',
                    len(dl_train), len(dl_valid), len(dl_test))
        for _ in tqdm.tqdm(dl_test, desc='gen-test-dl', ncols=80):
            pass

        num_node_type = len(g.node_type_encode)
        num_edge_type = len(g.edge_type_encode)
        logger.info('#node_type %d, #edge_type %d', num_node_type, num_edge_type)

        if conv_name != 'logi':
            if conv_name == '':
                gnn = None
            else:
                gnn = GNN(conv_name=conv_name,
                          n_in=num_feat,
                          n_hid=n_hid, n_heads=n_heads, n_layers=n_layers,
                          dropout=dropout,
                          num_node_type=num_node_type,
                          num_edge_type=num_edge_type
                          )

            model = Net(gnn, num_feat, num_embed=n_hid, n_hidden=n_hid)
        else:
            model = NetLogi(num_feat)
        model.to(device)

        model_loaded = False
        if continue_training:
            files = glob.glob(f'{dir_model}/model-{conv_name}-{seed}*')
            if len(files) > 0:
                files.sort(key=os.path.getmtime)
                load_file = files[-1]
                logger.info(f'Continue training from checkpoint {load_file}')
                model.load_state_dict(torch.load(load_file))
                model_loaded = True

        if optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters())
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters())
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        elif optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters())

        pb = partial(
            prepare_batch, g=g, fstore=store, ts_range=train_range,
            default_feature=np.zeros_like(x0))

        loss = nn.CrossEntropyLoss()

        from ignite.engine import EventEnum, _prepare_batch
        from ignite.distributed import utils as idist
        from ignite.engine.deterministic import DeterministicEngine
        if idist.has_xla_support:
            import torch_xla.core.xla_model as xm

        from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

        class ForwardEvents(EventEnum):
            FORWARD_STARTED = 'forward_started'
            FORWARD_COMPLETED = 'forward_completed'

        """create UDF trainer to register forward events"""
        def udf_supervised_trainer(
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                loss_fn: Union[Callable, torch.nn.Module],
                device: Optional[Union[str, torch.device]] = None,
                non_blocking: bool = False,
                prepare_batch: Callable = _prepare_batch,
                output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
                deterministic: bool = False,
        ) -> Engine:
            device_type = device.type if isinstance(device, torch.device) else device
            on_tpu = "xla" in device_type if device_type is not None else False

            if on_tpu and not idist.has_xla_support:
                raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

            def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
                model.train()
                optimizer.zero_grad()
                x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

                engine.fire_event(ForwardEvents.FORWARD_STARTED)
                y_pred = model(x)
                engine.fire_event(ForwardEvents.FORWARD_COMPLETED)

                loss = loss_fn(y_pred, y)
                loss.backward()

                if on_tpu:
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()

                return output_transform(x, y, y_pred, loss)

            trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)
            trainer.register_events(*ForwardEvents)

            return trainer

        trainer = udf_supervised_trainer(
            model, optimizer, loss,
            device=device, prepare_batch=pb)

        pb = partial(
            prepare_batch, g=g, fstore=store, ts_range=valid_range,
            default_feature=np.zeros_like(x0))
        evaluator = create_supervised_evaluator(model,
                                                metrics={
                                                    'accuracy': Accuracy(),
                                                    'loss': Loss(loss),
                                                    'ap': AveragePrecision(
                                                        output_transform=lambda out: (out[0][:, 1], out[1])),
                                                    'auc': ROC_AUC(
                                                        output_transform=lambda out: (out[0][:, 1], out[1])),
                                                }, device=device, prepare_batch=pb)

        if model_loaded:
            with torch.no_grad():
                evaluator.run(dl_test)
            metrics = evaluator.state.metrics
            logger.info(
                'Loaded model stat: Test\tLoss %.2f\tAccuracy %.2f\tAUC %.4f\tAP %.4f',
                metrics['loss'], metrics['accuracy'],
                metrics['auc'], metrics['ap']
            )

        scheduler = CosineAnnealingScheduler(
            optimizer, 'lr',
            start_value=0.05, end_value=1e-4,
            cycle_size=len(dl_train) * max_epochs)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        pbar_train = tqdm.tqdm(desc='train', total=len(dl_train), ncols=100)
        t_epoch = Timer(average=True)
        t_epoch.pause()

        t_iter = Timer(average=True)
        t_iter.pause()

        @trainer.on(ForwardEvents.FORWARD_STARTED)
        def resume_timer(engine):
            t_epoch.resume()
            t_iter.resume()

        @trainer.on(ForwardEvents.FORWARD_COMPLETED)
        def pause_timer(engine):
            t_epoch.pause()
            t_iter.pause()
            t_iter.step()

        @trainer.on(Events.EPOCH_STARTED)
        def log_training_loss(engine):
            pbar_train.refresh()
            pbar_train.reset()

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            pbar_train.update(1)
            pbar_train.set_description(
                'Train [Eopch %03d] Loss %.4f T-iter %.4f' % (
                    engine.state.epoch, engine.state.output, t_iter.value()
                )
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            t_epoch.step()
            evaluator.run(dl_valid)
            metrics = evaluator.state.metrics
            logger.info(
                '[Epoch %03d]\tLoss %.4f\tAccuracy %.4f\tAUC %.4f\tAP %.4f \tTime %.2f / %03d',
                engine.state.epoch,
                metrics['loss'], metrics['accuracy'],
                metrics['auc'], metrics['ap'],
                t_epoch.value(), t_epoch.step_count
            )
            t_iter.reset()
            t_epoch.pause()
            t_iter.pause()

        def score_function(engine):
            return engine.state.metrics['auc']

        handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, handler)

        cp = ModelCheckpoint(dir_model, f'model-{conv_name}-{seed}', n_saved=1,
                             create_dir=True,
                             score_function=lambda e: evaluator.state.metrics['auc'],
                             require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, cp, {conv_name: model})

        trainer.run(dl_train, max_epochs=max_epochs)

        path_model = cp.last_checkpoint
        model.load_state_dict(torch.load(path_model))
        model.eval()
        with torch.no_grad():
            evaluator.run(dl_test)
        metrics = evaluator.state.metrics
        logger.info(
            'Test\tLoss %.2f\tAccuracy %.2f\tAUC %.4f\tAP %.4f',
            metrics['loss'], metrics['accuracy'],
            metrics['auc'], metrics['ap']
        )

        stats.update(dict(metrics))

        stats['epoch'] = trainer.state.epoch,

        row = pd.DataFrame([stats])
        if os.path.exists(path_result):
            result = pd.read_csv(path_result)
        else:
            result = pd.DataFrame()
        result = result.append(row)
        result.to_csv(path_result, index=False)


if __name__ == '__main__':
    fire.Fire(main)
