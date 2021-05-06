from typing import List, Optional, Tuple, NamedTuple

import numpy as np
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import degree
from torch_geometric.data.sampler import Adj


class DegreeWeightedNeighborSampler(torch.utils.data.DataLoader):

    def __init__(self, 
                 node_type_weights: torch.FloatTensor, 
                 edge_index: torch.Tensor, sizes: List[int],
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 flow: str = "source_to_target", 
                 sample_base: int = 4, enable_degree_weight: bool = True, 
                 **kwargs):

        N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        edge_attr = torch.arange(edge_index.size(1))
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, (N, N),
                                           is_sorted=False)
        adj = adj.t() if flow == 'source_to_target' else adj
        self.adj = adj.to('cpu')

        if node_idx is None:
            node_idx = torch.arange(N)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero().view(-1)

        self.sizes = sizes
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.sample_base = sample_base
        self.node_weight = node_type_weights
        if enable_degree_weight:
            self.node_weight *= degree(edge_index[0], N)

        super(DegreeWeightedNeighborSampler, self).__init__(
            node_idx.tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        adjs: List[Adj] = []

        n_id = batch
        n_id_rval = n_id
        for size in self.sizes:
            n_id_prev = n_id
            size_prev = n_id_prev.shape[0]
            size_expected = size_prev * (size+1)

            adj, n_id = self.adj.sample_adj(n_id, size*self.sample_base, 
                                            replace=False)
            
            if size_expected < n_id.shape[0]:    
                assert not (n_id[:size_prev]!=n_id_prev).any()

                n_id_to_sample = n_id[size_prev:]
                score = self.node_weight[n_id_to_sample]
                score /= torch.sum(score)
                p = score.cpu().numpy()
                mask = np.random.choice(
                    int(n_id_to_sample.shape[0]), 
                    size_expected-size_prev, p=p, replace=False)

                mask = torch.cat([
                    torch.arange(size_prev).long(),
                    size_prev+torch.LongTensor(mask)]).to(n_id.device)
                adj = adj[:, mask]
                n_id = n_id[mask]
                assert n_id.shape == torch.unique(n_id).shape

            row, col, e_id = adj.coo()
            size = adj.sparse_sizes() 

            if self.flow == 'source_to_target':
                edge_index = torch.stack([col, row], dim=0)
                size = size[::-1]
            else:
                edge_index = torch.stack([row, col], dim=0)

            adjs.append(Adj(edge_index, e_id, size))

        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)
