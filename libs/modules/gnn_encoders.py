from overrides import overrides

import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F

from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_softmax


class GNNEncoder(torch.nn.Module, Registrable):
    """The base class for all GNN encoder modules.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError


@GNNEncoder.register("gcn")
class GCN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x.squeeze()

    def get_output_dim(self):
        return self.conv2.out_channels


@GNNEncoder.register("cross_gat")
class CrossGAT(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(CrossGAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim1)
        self.cross1 = GATConv(hidden_dim1, hidden_dim1)

        self.conv2 = GATConv(hidden_dim1, hidden_dim2)
        self.cross2 = GATConv(hidden_dim2, hidden_dim2)

        self.conv3 = GATConv(hidden_dim2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        cross_edge_index = data.edge_index_cross

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.cross1(x, cross_edge_index)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.cross2(x, cross_edge_index)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.conv3(x, edge_index)

        return x.squeeze()

    def get_output_dim(self):
        return self.conv3.out_channels


@GNNEncoder.register("cross_gcn")
class CrossGCN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(CrossGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.cross1 = GCNConv(hidden_dim1, hidden_dim1)

        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.cross2 = GCNConv(hidden_dim2, hidden_dim2)

        self.conv3 = GCNConv(hidden_dim2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        cross_edge_index = data.edge_index_cross

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.cross1(x, cross_edge_index)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.cross2(x, cross_edge_index)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.conv3(x, edge_index)

        return x.squeeze()

    def get_output_dim(self):
        return self.conv3.out_channels


@GNNEncoder.register("dual_gcn")
class DualGCN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DualGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        self.cross1 = GCNConv(input_dim, hidden_dim)
        self.cross2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        cross_edge_index = data.edge_index_cross

        # The dependency information
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1)
        x1 = self.conv2(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1)

        # Cross information
        x2 = self.cross1(x, cross_edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2)
        x2 = self.cross2(x2, cross_edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2)

        return torch.cat([x1.squeeze(), x2.squeeze()], -1)

    def get_output_dim(self):
        return self.conv2.out_channels * 2


@GNNEncoder.register("dual_gat")
class DualGAT(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DualGAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim)

        self.cross1 = GATConv(input_dim, hidden_dim)
        self.cross2 = GATConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        cross_edge_index = data.edge_index_cross

        # The dependency information
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1)
        x1 = self.conv2(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1)

        # Cross information
        x2 = self.cross1(x, cross_edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2)
        x2 = self.cross2(x2, cross_edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2)

        return torch.cat([x1.squeeze(), x2.squeeze()], -1)

    def get_output_dim(self):
        return self.conv2.out_channels * 2
