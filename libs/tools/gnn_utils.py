from typing import Tuple, List, Dict

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse


class GraphData(Data):
    def __init__(self, cross_edges, **kargs):
        super(GraphData, self).__init__(**kargs)
        self.edge_index_cross = cross_edges

    def __inc__(self, key, value):
        if key == 'edge_index_cross':
            return self.x.size(0)
        else:
            return super(GraphData, self).__inc__(key, value)


def allennlp_edges_to_ordinary(allennlp_edges):

    # Get rid of -1  (by default, no edge is indicated with -1)
    edges = allennlp_edges + 1
    dense_indices = (edges != 0).float()
    edges = edges * dense_indices
    # Dense to sparse
    indices, values = dense_to_sparse(edges)
    return indices, values


def create_torch_geometric_batch(
    # batch * num_tokens * embedding_dim
    embedded_tokens_batch: List[torch.LongTensor],
    edges_batch: List[torch.LongTensor],  # batch *
    cross_edges_batch: List[torch.LongTensor]  # batch *
):
    """
    """
    assert embedded_tokens_batch.size()[0] == edges_batch.size()[0]

    batch_data = []
    for tokens, edges, cross_edges in zip(embedded_tokens_batch, edges_batch, cross_edges_batch):

        indices, values = allennlp_edges_to_ordinary(edges)
        indices_cross, _ = allennlp_edges_to_ordinary(cross_edges)

        # Create pytorch geometric data
        data = GraphData(x=tokens, edge_index=indices,
                         edge_attr=values, cross_edges=indices_cross)
        batch_data.append(data)

    batch = Batch.from_data_list(batch_data)

    return batch


# def back_to_normal_torch_batch():
