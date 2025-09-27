import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network Encoder.

    This module takes a batch of molecular graphs and computes a graph-level embedding
    for each, based on the architecture defined in the design document.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.2):
        super().__init__()

        if num_layers < 2:
            raise ValueError("GCNEncoder must have at least 2 layers (1 hidden, 1 output).")

        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer (before pooling)
        self.conv_layers.append(GCNConv(hidden_dim, output_dim))

        # The design doc shows batch norm applied after each of the first 3 layers.
        # This matches the number of hidden layers.
        # A batch norm is added for the input layer as well for stability.
        self.batch_norms.insert(0, nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        """
        Forward pass for the GCN Encoder.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, input_dim].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            batch (torch.Tensor): Batch vector of shape [num_nodes], which assigns each node to a graph.

        Returns:
            torch.Tensor: A graph-level embedding of shape [batch_size, output_dim].
        """
        # Message Passing Layers
        for i in range(self.num_layers - 1):
            x = self.conv_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Final convolution before pooling
        x = self.conv_layers[-1](x, edge_index)

        # Global Pooling to get graph-level representation
        graph_repr = global_mean_pool(x, batch)

        return graph_repr

if __name__ == '__main__':
    # --- Example Usage & Test ---
    print("--- Testing GCNEncoder Layer ---")

    # Parameters from specification-doc.md (using a smaller input_dim for test)
    # The preprocessor currently creates 7 features, but the spec aims for 128.
    # The model should be flexible to handle either.
    input_dim = 7
    hidden_dim = 256
    output_dim = 256
    num_layers = 4
    dropout = 0.2

    # Create the encoder
    encoder = GCNEncoder(input_dim, hidden_dim, output_dim, num_layers, dropout)
    print("Encoder architecture:", encoder)
    encoder.train() # Set to training mode to activate dropout

    # --- Create a batch of dummy graph data ---
    # Simulate a batch of 2 graphs
    # Graph 1: 3 nodes, 2 edges
    x1 = torch.randn(3, input_dim)
    edge_index1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()
    data1 = Data(x=x1, edge_index=edge_index1)

    # Graph 2: 4 nodes, 3 edges
    x2 = torch.randn(4, input_dim)
    edge_index2 = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long).t().contiguous()
    data2 = Data(x=x2, edge_index=edge_index2)

    # Create a batch from the list of graphs
    batch = Batch.from_data_list([data1, data2])
    print("\nBatch object created for 2 graphs.")
    print(batch)

    # --- Forward Pass ---
    print("\nRunning forward pass...")
    output = encoder(batch.x, batch.edge_index, batch.batch)
    print("Input node features shape:", batch.x.shape)
    print("Output graph embedding shape:", output.shape)

    # The output should have shape [batch_size, output_dim]
    assert output.shape == (2, output_dim)
    print("Forward pass successful.")

    # --- Backward Pass (Gradient Check) ---
    print("\nRunning backward pass...")
    target = torch.randn(2, output_dim)
    loss = (output - target).pow(2).mean()
    loss.backward()

    # Check if gradients exist for the first convolutional layer's weights
    grad = encoder.conv_layers[0].lin.weight.grad
    print("Gradient shape for first conv layer:", grad.shape)
    assert grad is not None, "Gradient is None! Backward pass failed."
    assert grad.abs().sum() > 0, "Gradient is all zeros!"
    print("Backward pass successful. Gradients are flowing.")

    print("\n--- GCNEncoder test completed successfully! ---")