import math
from metrics import probit
import monotonicnetworks as lmn
import numpy as np
from sklearn.neighbors import NearestNeighbors
from spatial import *
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, knn_graph


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN).
    """

    def __init__(
        self,
        num_features_in: int = 6,
        num_features_out: int = 1,
        gnn_hidden_dim: int = 32,
        gnn_emb_dim: int = 32,
        k: int = 5,
        p_dropout: float = 0.5
    ) -> None:
        """
        Initialize a GCN model.

        Parameters
        ----------
        num_features_in : int, optional
            The dimensionality of the input features. Defaults to 6.
        num_features_out : int, optional
            The dimensionality of the model output. Defaults to 1.
        gnn_hidden_dim : int, optional
            The dimensionality of the hidden layers in the GCN. Defaults to 32.
        gnn_emb_dim : int, optional
            The dimensionality of the final embedding layer. Defaults to 32.
        k : int, optional
            Number of nearest neighbors for graph construction. Defaults to 5.
        p_dropout : float, optional
            Dropout probability. Defaults to 0.5.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.p_dropout = p_dropout

        self.conv1 = GCNConv(num_features_in, gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_emb_dim)
        self.fc = nn.Linear(gnn_emb_dim, num_features_out)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ei: torch.Tensor | None,
        ew: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Forward pass of the GCN.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape [num_nodes, num_features_in].
        c : torch.Tensor
            Node coordinates, shape [num_nodes, coordinate_dim].
        ei : torch.Tensor | None
            Precomputed edge indices (if provided).
        ew : torch.Tensor | None
            Precomputed edge weights (if provided).

        Returns
        -------
        torch.Tensor
            Model output, shape [num_nodes, num_features_out].
        """
        x = x.float().to(self.device)
        c = c.float().to(self.device)

        if torch.is_tensor(ei) and torch.is_tensor(ew):
            edge_index = ei.to(self.device)
            edge_weight = ew.to(self.device)
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            edge_weight = makeEdgeWeight(c, edge_index).to(self.device)

        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, self.p_dropout, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, self.p_dropout, training=self.training)
        output = self.fc(h2)
        return output


class GAT(nn.Module):
    """
    Graph Attention Network (GAT).
    """

    def __init__(
        self,
        num_features_in: int = 6,
        num_features_out: int = 1,
        gnn_hidden_dim: int = 32,
        gnn_emb_dim: int = 32,
        k: int = 5,
        p_dropout: float = 0.5
    ) -> None:
        """
        Initialize a GAT model.

        Parameters
        ----------
        num_features_in : int, optional
            The dimensionality of the input features. Defaults to 6.
        num_features_out : int, optional
            The dimensionality of the model output. Defaults to 1.
        gnn_hidden_dim : int, optional
            The dimensionality of the hidden layers in the GAT. Defaults to 32.
        gnn_emb_dim : int, optional
            The dimensionality of the final embedding layer. Defaults to 32.
        k : int, optional
            Number of nearest neighbors for graph construction. Defaults to 5.
        p_dropout : float, optional
            Dropout probability. Defaults to 0.5.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.p_dropout = p_dropout

        self.conv1 = GATConv(num_features_in, gnn_hidden_dim)
        self.conv2 = GATConv(gnn_hidden_dim, gnn_emb_dim)
        self.fc = nn.Linear(gnn_emb_dim, num_features_out)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ei: torch.Tensor | None,
        ew: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Forward pass of the GAT.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape [num_nodes, num_features_in].
        c : torch.Tensor
            Node coordinates, shape [num_nodes, coordinate_dim].
        ei : torch.Tensor | None
            Precomputed edge indices (if provided).
        ew : torch.Tensor | None
            Precomputed edge weights (if provided).

        Returns
        -------
        torch.Tensor
            Model output, shape [num_nodes, num_features_out].
        """
        x = x.float().to(self.device)
        c = c.float().to(self.device)

        if torch.is_tensor(ei) and torch.is_tensor(ew):
            edge_index = ei.to(self.device)
            edge_weight = ew.to(self.device)
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            edge_weight = makeEdgeWeight(c, edge_index).to(self.device)

        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, self.p_dropout, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, self.p_dropout, training=self.training)
        output = self.fc(h2)
        return output


class GSAGE(nn.Module):
    """
    GraphSAGE (SAmple and aggreGatE).
    """

    def __init__(
        self,
        num_features_in: int = 6,
        num_features_out: int = 1,
        gnn_hidden_dim: int = 32,
        gnn_emb_dim: int = 32,
        k: int = 5,
        p_dropout: float = 0.5
    ) -> None:
        """
        Initialize a GraphSAGE model.

        Parameters
        ----------
        num_features_in : int, optional
            The dimensionality of the input features. Defaults to 6.
        num_features_out : int, optional
            The dimensionality of the model output. Defaults to 1.
        gnn_hidden_dim : int, optional
            The dimensionality of the hidden layers in the GraphSAGE model.
            Defaults to 32.
        gnn_emb_dim : int, optional
            The dimensionality of the final embedding layer. Defaults to 32.
        k : int, optional
            Number of nearest neighbors for graph construction. Defaults to 5.
        p_dropout : float, optional
            Dropout probability. Defaults to 0.5.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.p_dropout = p_dropout

        self.conv1 = SAGEConv(num_features_in, gnn_hidden_dim)
        self.conv2 = SAGEConv(gnn_hidden_dim, gnn_emb_dim)
        self.fc = nn.Linear(gnn_emb_dim, num_features_out)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ei: torch.Tensor | None,
        ew: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Forward pass of the GraphSAGE model.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape [num_nodes, num_features_in].
        c : torch.Tensor
            Node coordinates, shape [num_nodes, coordinate_dim].
        ei : torch.Tensor | None
            Precomputed edge indices (if provided).
        ew : torch.Tensor | None
            Precomputed edge weights (if provided). 
            Note: GraphSAGE layer here does not use edge weights explicitly,
            but we keep the parameter for consistency.

        Returns
        -------
        torch.Tensor
            Model output, shape [num_nodes, num_features_out].
        """
        x = x.float().to(self.device)
        c = c.float().to(self.device)

        if torch.is_tensor(ei) and torch.is_tensor(ew):
            edge_index = ei.to(self.device)
            # Edge weights are not directly used in SAGEConv here
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            _ = makeEdgeWeight(c, edge_index).to(self.device)  # not used directly

        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, self.p_dropout, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, self.p_dropout, training=self.training)
        output = self.fc(h2)
        return output


class LossWrapperGNN(nn.Module):
    """
    A wrapper class for computing loss on GNN outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: str = "mse",
        k: int = 5,
        batch_size: int = 2048
    ) -> None:
        """
        Initialize the loss wrapper with a specified model and criterion.

        Parameters
        ----------
        model : nn.Module
            The GNN model to be wrapped.
        loss : str, optional
            The type of loss function to use. Choices are ["mse", "l1"].
            Defaults to "mse".
        k : int, optional
            Number of nearest neighbors for graph construction (not used here,
            but stored if needed). Defaults to 5.
        batch_size : int, optional
            Batch size for training or evaluation (not enforced here, but stored).
            Defaults to 2048.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.k = k
        self.batch_size = batch_size

        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError("Loss function not recognized. Choose from ['mse', 'l1'].")

    def forward(
        self,
        input_data: torch.Tensor,
        targets: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_weight: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Compute the loss for a given batch of data.

        Parameters
        ----------
        input_data : torch.Tensor
            Node features, shape [num_nodes, num_features_in].
        targets : torch.Tensor
            Ground truth targets, shape [num_nodes].
        coords : torch.Tensor
            Node coordinates, shape [num_nodes, coordinate_dim].
        edge_index : torch.Tensor | None
            Precomputed edge indices (if provided).
        edge_weight : torch.Tensor | None
            Precomputed edge weights (if provided).

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        outputs = self.model(input_data, coords, edge_index, edge_weight)
        loss_value = self.criterion(
            targets.float().view(-1), outputs.float().view(-1)
        )
        return loss_value


def _cal_freq_list(
    freq_init: str,
    frequency_num: int,
    max_radius: float,
    min_radius: float
) -> np.ndarray:
    """
    Compute a list of frequencies based on the initialization strategy.

    Parameters
    ----------
    freq_init : str
        Either "random" or "geometric".
    frequency_num : int
        Number of frequencies to generate.
    max_radius : float
        Maximum radius for geometric initialization.
    min_radius : float
        Minimum radius for geometric initialization.

    Returns
    -------
    np.ndarray
        An array of frequencies.
    """
    if freq_init == "random":
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = math.log(float(max_radius) / float(min_radius)) / (
            float(frequency_num) - 1.0
        )
        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment
        )
        freq_list = 1.0 / timescales
    else:
        raise ValueError("freq_init must be either 'random' or 'geometric'.")
    return freq_list


def get_activation_function(
    activation: str,
    context_str: str
) -> nn.Module:
    """
    Return a PyTorch activation function.

    Parameters
    ----------
    activation : str
        Name of the activation function. Options: ['leakyrelu', 'relu',
        'sigmoid', 'tanh'].
    context_str : str
        Identifier used for error messages.

    Returns
    -------
    nn.Module
        The requested activation function.

    Raises
    ------
    Exception
        If the specified activation function is not recognized.
    """
    activation = activation.lower()
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise Exception(f"{context_str} activation '{activation}' not recognized.")


class LayerNorm(nn.Module):
    """
    A simple layer normalization module.
    """

    def __init__(
        self,
        feature_dim: int,
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta", self.beta)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for layer normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, feature_dim].

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as `x`.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SingleFeedForwardNN(nn.Module):
    """
    A single fully-connected (linear) layer with optional activation,
    dropout, layer normalization, and skip connection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: float | None = None,
        activation: str = "sigmoid",
        use_layernormalize: bool = False,
        skip_connection: bool = False,
        context_str: str = ""
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate is not None else None
        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # Skip connection is only valid if input and output dimensions match
        self.skip_connection = skip_connection and (self.input_dim == self.output_dim)

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape [batch_size, ..., input_dim].

        Returns
        -------
        torch.Tensor
            Output of shape [batch_size, ..., output_dim].
        """
        assert input_tensor.size(-1) == self.input_dim

        # Linear transformation
        output = self.linear(input_tensor)

        # Activation
        output = self.act(output)

        # Dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # Skip connection
        if self.skip_connection:
            output = output + input_tensor

        # Layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output


class MultiLayerFeedForwardNN(nn.Module):
    """
    A multi-layer feed-forward neural network consisting of:
    - N hidden layers (with optional activation, layer norm, dropout, skip connections)
    - A final layer without extra transformations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int = 0,
        dropout_rate: float = 0.5,
        hidden_dim: int = -1,
        activation: str = "relu",
        use_layernormalize: bool = True,
        skip_connection: bool = False,
        context_str: str | None = None
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str if context_str else ""

        self.layers = nn.ModuleList()

        # If no hidden layers, just build a single-layer network
        if self.num_hidden_layers <= 0:
            self.layers.append(
                SingleFeedForwardNN(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_layernormalize=False,
                    skip_connection=False,
                    context_str=self.context_str
                )
            )
        else:
            # First hidden layer
            self.layers.append(
                SingleFeedForwardNN(
                    input_dim=self.input_dim,
                    output_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_layernormalize=self.use_layernormalize,
                    skip_connection=self.skip_connection,
                    context_str=self.context_str
                )
            )

            # Middle hidden layers
            for _ in range(self.num_hidden_layers - 1):
                self.layers.append(
                    SingleFeedForwardNN(
                        input_dim=self.hidden_dim,
                        output_dim=self.hidden_dim,
                        dropout_rate=self.dropout_rate,
                        activation=self.activation,
                        use_layernormalize=self.use_layernormalize,
                        skip_connection=self.skip_connection,
                        context_str=self.context_str
                    )
                )

            # Final layer
            self.layers.append(
                SingleFeedForwardNN(
                    input_dim=self.hidden_dim,
                    output_dim=self.output_dim,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_layernormalize=False,
                    skip_connection=False,
                    context_str=self.context_str
                )
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape [batch_size, ..., input_dim].

        Returns
        -------
        torch.Tensor
            Output of shape [batch_size, ..., output_dim].
        """
        assert input_tensor.size(-1) == self.input_dim

        output = input_tensor
        for layer in self.layers:
            output = layer(output)

        return output


class GridCellSpatialRelationEncoder(nn.Module):
    """
    Encodes spatial coordinates using sinusoidal functions across multiple frequencies
    (akin to positional encodings).
    """

    def __init__(
        self,
        spa_embed_dim: int,
        coord_dim: int = 2,
        frequency_num: int = 16,
        max_radius: float = 0.01,
        min_radius: float = 0.00001,
        freq_init: str = "geometric",
        ffn: bool | None = None
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn

        # Generate frequency list and matrix
        self.freq_list = _cal_freq_list(
            self.freq_init, self.frequency_num, self.max_radius, self.min_radius
        )
        self._cal_freq_mat()
        self.input_embed_dim = self.coord_dim * self.frequency_num * 2

        # Optionally create a multi-layer feed-forward net to transform
        # the sinusoidal embeddings to the final embedding.
        if self.ffn is not None:
            self.ffn = MultiLayerFeedForwardNN(
                input_dim=self.input_embed_dim,
                output_dim=self.spa_embed_dim
            )

    def _cal_freq_mat(self) -> None:
        """
        Internal helper to build a repeated matrix of frequencies.
        """
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        self.freq_mat = np.repeat(freq_mat, 2, axis=1)  # shape: (frequency_num, 2)

    def _make_input_embeds(self, coords: list | np.ndarray) -> np.ndarray:
        """
        Build raw sinusoidal embeddings for each coordinate.

        Parameters
        ----------
        coords : list | np.ndarray
            Shape (batch_size, num_context_points, coord_dim).

        Returns
        -------
        np.ndarray
            Encoded sine/cosine features with shape:
            (batch_size, num_context_points, input_embed_dim).
        """
        if isinstance(coords, np.ndarray):
            assert coords.shape[2] == self.coord_dim
        elif isinstance(coords, list):
            assert len(coords[0][0]) == self.coord_dim
        else:
            raise TypeError("coords must be a list or np.ndarray.")

        coords_mat = np.asarray(coords, dtype=float)  # (batch_size, num_context_pt, coord_dim)
        batch_size, num_context_pt, _ = coords_mat.shape

        # Expand dims to match frequency_num and 2 (sin/cos) multiplication
        coords_mat = np.expand_dims(coords_mat, axis=3)  # add dim for frequency
        coords_mat = np.expand_dims(coords_mat, axis=4)  # add dim for sin/cos
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)
        coords_mat = np.repeat(coords_mat, 2, axis=4)

        # Multiply by freq_mat to shift frequencies
        spr_embeds = coords_mat * self.freq_mat

        # sin for even indices, cos for odd indices
        spr_embeds[..., 0::2] = np.sin(spr_embeds[..., 0::2])
        spr_embeds[..., 1::2] = np.cos(spr_embeds[..., 1::2])

        # Reshape to (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = np.reshape(
            spr_embeds,
            (batch_size, num_context_pt, -1)
        )
        return spr_embeds

    def forward(self, coords: list | np.ndarray) -> torch.Tensor:
        """
        Convert spatial coords into sinusoidal embeddings, then optionally
        pass them through a feed-forward network.

        Parameters
        ----------
        coords : list | np.ndarray
            Shape (batch_size, num_context_pt, coord_dim).

        Returns
        -------
        torch.Tensor
            If ffn is provided, shape (batch_size, num_context_pt, spa_embed_dim);
            otherwise shape (batch_size, num_context_pt, input_embed_dim).
        """
        spr_embeds_np = self._make_input_embeds(coords)
        spr_embeds_torch = torch.FloatTensor(spr_embeds_np).to(self.device)

        if self.ffn is not None:
            return self.ffn(spr_embeds_torch)
        return spr_embeds_torch


class PEGCN(nn.Module):
    """
    A GCN model that integrates a positional (grid cell) encoder.
    Optionally supports an auxiliary task for Moran's I prediction (MAT).
    """

    def __init__(
        self,
        num_features_in: int = 6,
        num_features_out: int = 1,
        gnn_hidden_dim: int = 32,
        gnn_emb_dim: int = 32,
        pe_hidden_dim: int = 128,
        pe_emb_dim: int = 64,
        k: int = 5,
        p_dropout: float = 0.5,
        MAT: bool = False
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_emb_dim = gnn_emb_dim
        self.pe_hidden_dim = pe_hidden_dim
        self.pe_emb_dim = pe_emb_dim
        self.k = k
        self.p_dropout = p_dropout
        self.MAT = MAT

        # Positional encoder
        self.spenc = GridCellSpatialRelationEncoder(
            spa_embed_dim=pe_hidden_dim,
            ffn=True,
            min_radius=1e-6,
            max_radius=360
        )
        # Additional MLP to compress the encoded embedding
        self.dec = nn.Sequential(
            nn.Linear(pe_hidden_dim, pe_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 2, pe_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 4, pe_emb_dim),
        )

        # GCN layers
        self.conv1 = GCNConv(num_features_in + pe_emb_dim, gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_emb_dim)

        # Final linear layer(s)
        self.fc = nn.Linear(gnn_emb_dim, num_features_out)
        if self.MAT:
            self.fc_morans = nn.Linear(gnn_emb_dim, num_features_out)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ei: torch.Tensor | None,
        ew: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PEGCN.

        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, num_features_in].
        c : torch.Tensor
            Node coordinates [num_nodes, coord_dim].
        ei : torch.Tensor | None
            Edge indices if precomputed.
        ew : torch.Tensor | None
            Edge weights if precomputed.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Single output if MAT=False, else a tuple (prediction, morans_prediction).
        """
        x = x.float().to(self.device)
        c = c.float().to(self.device)

        # Build or use precomputed edges
        if torch.is_tensor(ei) and torch.is_tensor(ew):
            edge_index = ei
            edge_weight = ew
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            edge_weight = makeEdgeWeight(c, edge_index).to(self.device)

        # Positional encoding
        c_reshaped = c.reshape(1, c.shape[0], c.shape[1])
        emb = self.spenc(c_reshaped.detach().cpu().numpy())  # shape [1, num_nodes, pe_hidden_dim]
        emb = emb.reshape(emb.shape[1], emb.shape[2])  # [num_nodes, pe_hidden_dim]
        emb = self.dec(emb).float().to(self.device)

        # Concatenate positional embeddings
        x = torch.cat((x, emb), dim=1)

        # GCN forward
        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, self.p_dropout, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, self.p_dropout, training=self.training)

        # Predictions
        output = self.fc(h2)
        if self.MAT:
            morans_output = self.fc_morans(h2)
            return output, morans_output
        return output


class PEGAT(nn.Module):
    """
    A GAT model that integrates a positional (grid cell) encoder.
    Optionally supports an auxiliary task for Moran's I prediction (MAT).
    """

    def __init__(
        self,
        num_features_in: int = 6,
        num_features_out: int = 1,
        gnn_hidden_dim: int = 32,
        gnn_emb_dim: int = 32,
        pe_hidden_dim: int = 128,
        pe_emb_dim: int = 64,
        k: int = 5,
        p_dropout: float = 0.5,
        MAT: bool = False
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_emb_dim = gnn_emb_dim
        self.pe_hidden_dim = pe_hidden_dim
        self.pe_emb_dim = pe_emb_dim
        self.k = k
        self.p_dropout = p_dropout
        self.MAT = MAT

        self.spenc = GridCellSpatialRelationEncoder(
            spa_embed_dim=pe_hidden_dim,
            ffn=True,
            min_radius=1e-6,
            max_radius=360
        )
        self.dec = nn.Sequential(
            nn.Linear(pe_hidden_dim, pe_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 2, pe_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 4, pe_emb_dim),
        )

        self.conv1 = GATConv(num_features_in + pe_emb_dim, gnn_hidden_dim)
        self.conv2 = GATConv(gnn_hidden_dim, gnn_emb_dim)

        self.fc = nn.Linear(gnn_emb_dim, num_features_out)
        if self.MAT:
            self.fc_morans = nn.Linear(gnn_emb_dim, num_features_out)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ei: torch.Tensor | None,
        ew: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = x.float().to(self.device)
        c = c.float().to(self.device)

        if torch.is_tensor(ei) and torch.is_tensor(ew):
            edge_index = ei
            edge_weight = ew
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            edge_weight = makeEdgeWeight(c, edge_index).to(self.device)

        c_reshaped = c.reshape(1, c.shape[0], c.shape[1])
        emb = self.spenc(c_reshaped.detach().cpu().numpy())
        emb = emb.reshape(emb.shape[1], emb.shape[2])
        emb = self.dec(emb).float().to(self.device)

        x = torch.cat((x, emb), dim=1)

        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, self.p_dropout, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, self.p_dropout, training=self.training)

        output = self.fc(h2)
        if self.MAT:
            morans_output = self.fc_morans(h2)
            return output, morans_output
        return output


class PEGSAGE(nn.Module):
    """
    A GraphSAGE model that integrates a positional (grid cell) encoder.
    Optionally supports an auxiliary task for Moran's I prediction (MAT).
    """

    def __init__(
        self,
        num_features_in: int = 6,
        num_features_out: int = 1,
        gnn_hidden_dim: int = 32,
        gnn_emb_dim: int = 32,
        pe_hidden_dim: int = 128,
        pe_emb_dim: int = 64,
        k: int = 5,
        p_dropout: float = 0.5,
        MAT: bool = False
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_emb_dim = gnn_emb_dim
        self.pe_hidden_dim = pe_hidden_dim
        self.pe_emb_dim = pe_emb_dim
        self.k = k
        self.p_dropout = p_dropout
        self.MAT = MAT

        self.spenc = GridCellSpatialRelationEncoder(
            spa_embed_dim=pe_hidden_dim,
            ffn=True,
            min_radius=1e-6,
            max_radius=360
        )
        self.dec = nn.Sequential(
            nn.Linear(pe_hidden_dim, pe_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 2, pe_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 4, pe_emb_dim),
        )

        self.conv1 = SAGEConv(num_features_in + pe_emb_dim, gnn_hidden_dim)
        self.conv2 = SAGEConv(gnn_hidden_dim, gnn_emb_dim)

        self.fc = nn.Linear(gnn_emb_dim, num_features_out)
        if self.MAT:
            self.fc_morans = nn.Linear(gnn_emb_dim, num_features_out)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ei: torch.Tensor | None,
        ew: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = x.float().to(self.device)
        c = c.float().to(self.device)

        if torch.is_tensor(ei) and torch.is_tensor(ew):
            edge_index = ei
            # Edge weights are not directly used in SAGEConv here
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            _ = makeEdgeWeight(c, edge_index).to(self.device)  # not used directly

        c_reshaped = c.reshape(1, c.shape[0], c.shape[1])
        emb = self.spenc(c_reshaped.detach().cpu().numpy())
        emb = emb.reshape(emb.shape[1], emb.shape[2])
        emb = self.dec(emb).float().to(self.device)

        x = torch.cat((x, emb), dim=1)

        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, self.p_dropout, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, self.p_dropout, training=self.training)

        output = self.fc(h2)
        if self.MAT:
            morans_output = self.fc_morans(h2)
            return output, morans_output
        return output


class LossWrapperPEGNN(nn.Module):
    """
    A wrapper that handles single- or multi-task loss computation for the
    positional-encoding GNN models (PEGCN, PEGAT, PEGSAGE).
    """

    def __init__(
        self,
        model: nn.Module,
        loss: str = "mse",
        k: int = 5,
        batch_size: int = 2048,
        task_num: int = 1,
        uw: bool = False,
        lamb: float = 0.0
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.k = k
        self.batch_size = batch_size
        self.task_num = task_num
        self.uw = uw
        self.lamb = lamb

        if task_num > 1:
            # For uncertainty weighting, we keep a log variance per task
            self.log_vars = nn.Parameter(torch.zeros((task_num)))

        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError("Loss function must be either 'mse' or 'l1'.")

    def forward(
        self,
        input_data: torch.Tensor,
        targets: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
        morans_input: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
        """
        Compute loss for single or multi-task outputs.

        Parameters
        ----------
        input_data : torch.Tensor
            Node features.
        targets : torch.Tensor
            Primary task ground-truth values.
        coords : torch.Tensor
            Node coordinates.
        edge_index : torch.Tensor | None
            Precomputed edge indices (if any).
        edge_weight : torch.Tensor | None
            Precomputed edge weights (if any).
        morans_input : torch.Tensor | None
            Optional precomputed target for the Moran's I auxiliary task.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, list[float]]
            If task_num=1, returns a single loss tensor.
            If task_num=2 and uw=True, returns (loss, log_vars.data).
            Otherwise, returns just the combined loss tensor.
        """
        # Single-task scenario
        if self.task_num == 1:
            outputs = self.model(input_data, coords, edge_index, edge_weight)
            loss_val = self.criterion(targets.float().reshape(-1), outputs.float().reshape(-1))
            return loss_val

        # Multi-task scenario
        outputs1, outputs2 = self.model(input_data, coords, edge_index, edge_weight)
        if torch.is_tensor(morans_input):
            targets2 = morans_input
        else:
            # If no Moran's input is provided, compute from the adjacency
            adj_matrix = knn_to_adj(
                knn_graph(coords, k=self.k),
                self.batch_size
            )
            with torch.enable_grad():
                targets2 = lw_tensor_local_moran(
                    targets,
                    sparse.csr_matrix(adj_matrix)
                ).to(self.device)

        if self.uw:
            # Uncertainty-weighted scenario
            precision1 = 0.5 * torch.exp(-self.log_vars[0])
            loss1 = self.criterion(targets.float().reshape(-1), outputs1.float().reshape(-1))
            loss1 = torch.sum(precision1 * loss1 + self.log_vars[0], dim=-1)

            precision2 = 0.5 * torch.exp(-self.log_vars[1])
            loss2 = self.criterion(targets2.float().reshape(-1), outputs2.float().reshape(-1))
            loss2 = torch.sum(precision2 * loss2 + self.log_vars[1], dim=-1)

            loss = loss1 + loss2
            loss = torch.mean(loss)
            return loss, self.log_vars.data.tolist()
        else:
            # Standard multi-task: weighted sum
            loss1 = self.criterion(targets.float().reshape(-1), outputs1.float().reshape(-1))
            loss2 = self.criterion(targets2.float().reshape(-1), outputs2.float().reshape(-1))
            loss = loss1 + self.lamb * loss2
            return loss


def compute_ybar(
    coords: torch.Tensor,
    y: np.ndarray | torch.Tensor,
    k: int
) -> np.ndarray:
    """
    Compute the local average (y-bar) for each point based on its k-nearest neighbors,
    using haversine distance.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinate tensor of shape [num_points, coord_dim] (e.g., [num_points, 2] for lat/lon).
        Values should be in degrees.
    y : np.ndarray | torch.Tensor
        Values (e.g., target variable) associated with each point.
    k : int
        Number of neighbors to use for computing y-bar.

    Returns
    -------
    np.ndarray
        An array of local averages (y-bar), one for each point.
    """
    # Convert coords from degrees to radians
    coords_rad = torch.deg2rad(coords)
    # If coords_rad is a torch.Tensor, convert to a NumPy array for sklearn
    coords_rad_np = coords_rad.cpu().numpy() if isinstance(coords_rad, torch.Tensor) else coords_rad
    # If y is a torch.Tensor, convert it to a NumPy array
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="haversine")
    nbrs.fit(coords_rad_np)

    # distances are not used, but kneighbors returns them by default
    _, indices = nbrs.kneighbors(coords_rad_np)

    # Exclude the first neighbor (the point itself) when computing the mean
    ybar = np.array([y_np[neighbors[1:]].mean() for neighbors in indices])
    return ybar


class PEGQCN(nn.Module):
    """
    GCN with a positional encoder, quantile regression (as proposed by Si, 2020),
    and optional ybar input.
    """

    def __init__(
        self,
        num_features_in: int = 6,
        num_features_out: int = 1,
        gnn_hidden_dim: int = 32,
        gnn_emb_dim: int = 32,
        pe_hidden_dim: int = 128,
        pe_emb_dim: int = 64,
        final_emb_dim: int = 8,
        k: int = 5,
        p_dropout: float = 0.5,
        MAT: bool = False,
        KNN: bool = True
    ) -> None:
        """
        Initialize the PEGQCN model.

        Parameters
        ----------
        num_features_in : int, optional
            Number of input features for the GCN, by default 6.
        num_features_out : int, optional
            Number of output features (e.g., for quantile regression), by default 1.
        gnn_hidden_dim : int, optional
            Dimension of the GCN hidden layer, by default 32.
        gnn_emb_dim : int, optional
            Dimension of the GCN embedding layer, by default 32.
        pe_hidden_dim : int, optional
            Dimension of the spatial encoder hidden layer, by default 128.
        pe_emb_dim : int, optional
            Dimension of the spatial encoder embedding layer, by default 64.
        final_emb_dim : int, optional
            Dimension of the final merged embedding, by default 8.
        k : int, optional
            Number of nearest neighbors for the KNN graph, by default 5.
        p_dropout : float, optional
            Dropout probability, by default 0.5.
        MAT : bool, optional
            If True, enable an auxiliary task for Moran's I, by default False.
        KNN : bool, optional
            If True, include `ybar` as an additional input to the monotonic subnet.
            By default True.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_emb_dim = gnn_emb_dim
        self.pe_hidden_dim = pe_hidden_dim
        self.pe_emb_dim = pe_emb_dim
        self.final_emb_dim = final_emb_dim
        self.k = k
        self.p_dropout = p_dropout
        self.MAT = MAT
        self.KNN = KNN

        # GCN layers
        self.conv1 = GCNConv(num_features_in, gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_emb_dim)

        # Spatial encoder
        self.spenc = GridCellSpatialRelationEncoder(
            spa_embed_dim=pe_hidden_dim,
            ffn=True,
            min_radius=1e-6,
            max_radius=360
        )
        self.dec_pe = nn.Sequential(
            nn.Linear(pe_hidden_dim, pe_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 2, pe_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 4, pe_emb_dim)
        )

        # Merge GCN and positional embeddings
        self.dec = nn.Sequential(
            nn.Linear(pe_emb_dim + gnn_emb_dim, final_emb_dim * 4),
            nn.Tanh(),
            nn.Linear(final_emb_dim * 4, final_emb_dim * 2),
            nn.Tanh(),
            nn.Linear(final_emb_dim * 2, final_emb_dim)
        )

        # Monotonic constraints setup
        if KNN:
            in_dim = final_emb_dim + 2  # includes tau and ybar
            monotonic_constraints = [0] * final_emb_dim + [1, 0]
        else:
            in_dim = final_emb_dim + 1  # only includes tau
            monotonic_constraints = [0] * final_emb_dim + [1]

        net = nn.Sequential(
            lmn.LipschitzLinear(in_dim, 32, kind="one-inf"),
            lmn.GroupSort(2),
            lmn.LipschitzLinear(32, num_features_out, kind="inf")
        )

        # Monotonic network for quantile regression
        self.monotonic_subnet = lmn.MonotonicWrapper(
            lipschitz_module=net,
            lipschitz_const=1.0,
            monotonic_constraints=monotonic_constraints
        )

        # Optional auxiliary task for Moran's I
        if MAT:
            self.fc_morans = lmn.LipschitzLinear(
                final_emb_dim, num_features_out, kind="inf"
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ei: torch.Tensor | None,
        ew: torch.Tensor | None,
        tau: torch.Tensor,
        ybar: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the PEGQCN model.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape [num_nodes, num_features_in].
        c : torch.Tensor
            Node coordinates, shape [num_nodes, coord_dim].
        ei : torch.Tensor | None
            Edge indices if precomputed. If None, KNN will be constructed on the fly.
        ew : torch.Tensor | None
            Edge weights if precomputed. If None, they will be computed on the fly.
        tau : torch.Tensor
            Quantile levels for each node, shape [num_nodes].
        ybar : torch.Tensor
            Optional additional feature (e.g., local average or uncertainty),
            shape [num_nodes].

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If MAT is False, returns the quantile regression output. If MAT is True,
            returns (quantile output, Moran's I output).
        """
        x = x.float().to(self.device)
        c = c.float().to(self.device)
        tau = tau.float().to(self.device)
        ybar = ybar.float().to(self.device)

        # Handle graph edges
        if torch.is_tensor(ei) and torch.is_tensor(ew):
            edge_index = ei
            edge_weight = ew
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            edge_weight = makeEdgeWeight(c, edge_index).to(self.device)

        # GCN forward pass
        x_emb = F.relu(self.conv1(x, edge_index, edge_weight))
        x_emb = F.dropout(x_emb, self.p_dropout, training=self.training)
        x_emb = F.relu(self.conv2(x_emb, edge_index, edge_weight))
        x_emb = F.dropout(x_emb, self.p_dropout, training=self.training)

        # Positional encoder forward pass
        c_reshaped = c.reshape(1, c.shape[0], c.shape[1])  # shape [1, num_nodes, coord_dim]
        c_emb = self.spenc(c_reshaped.detach().cpu().numpy())  # shape [1, num_nodes, pe_hidden_dim]
        c_emb = c_emb.reshape(c_emb.shape[1], c_emb.shape[2])
        c_emb = self.dec_pe(c_emb).float().to(self.device)

        # Merge GCN and positional embeddings
        l_emb = torch.cat((c_emb, x_emb), dim=1)
        phi_emb = self.dec(l_emb).float()

        # Build monotonic input
        tau = tau.view(-1, 1)
        phi_til_emb = torch.cat((phi_emb, tau), dim=1)
        if self.KNN:
            ybar = ybar.view(-1, 1)
            phi_til_emb = torch.cat((phi_til_emb, ybar), dim=1)

        # Monotonic regression output (quantile)
        output = self.monotonic_subnet(phi_til_emb)

        # Auxiliary task (Moran's I) if enabled
        if self.MAT:
            morans_output = self.fc_morans(phi_emb)
            return output, morans_output
        return output


class PEGQAT(nn.Module):
    """
    GAT with a positional encoder, quantile regression (as proposed by Si, 2020),
    and optional ybar input.
    """

    def __init__(
        self,
        num_features_in: int = 6,
        num_features_out: int = 1,
        gnn_hidden_dim: int = 32,
        gnn_emb_dim: int = 32,
        pe_hidden_dim: int = 128,
        pe_emb_dim: int = 64,
        final_emb_dim: int = 8,
        k: int = 5,
        p_dropout: float = 0.5,
        MAT: bool = False,
        KNN: bool = True
    ) -> None:
        """
        Initialize the PEGQCN model.

        Parameters
        ----------
        num_features_in : int, optional
            Number of input features for the GCN, by default 6.
        num_features_out : int, optional
            Number of output features (e.g., for quantile regression), by default 1.
        gnn_hidden_dim : int, optional
            Dimension of the GCN hidden layer, by default 32.
        gnn_emb_dim : int, optional
            Dimension of the GCN embedding layer, by default 32.
        pe_hidden_dim : int, optional
            Dimension of the spatial encoder hidden layer, by default 128.
        pe_emb_dim : int, optional
            Dimension of the spatial encoder embedding layer, by default 64.
        final_emb_dim : int, optional
            Dimension of the final merged embedding, by default 8.
        k : int, optional
            Number of nearest neighbors for the KNN graph, by default 5.
        p_dropout : float, optional
            Dropout probability, by default 0.5.
        MAT : bool, optional
            If True, enable an auxiliary task for Moran's I, by default False.
        KNN : bool, optional
            If True, include `ybar` as an additional input to the monotonic subnet.
            By default True.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_emb_dim = gnn_emb_dim
        self.pe_hidden_dim = pe_hidden_dim
        self.pe_emb_dim = pe_emb_dim
        self.final_emb_dim = final_emb_dim
        self.k = k
        self.p_dropout = p_dropout
        self.MAT = MAT
        self.KNN = KNN

        # GAT layers
        self.conv1 = GATConv(num_features_in, gnn_hidden_dim)
        self.conv2 = GATConv(gnn_hidden_dim, gnn_emb_dim)

        # Spatial encoder
        self.spenc = GridCellSpatialRelationEncoder(
            spa_embed_dim=pe_hidden_dim,
            ffn=True,
            min_radius=1e-6,
            max_radius=360
        )
        self.dec_pe = nn.Sequential(
            nn.Linear(pe_hidden_dim, pe_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 2, pe_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 4, pe_emb_dim)
        )

        # Merge GAT and positional embeddings
        self.dec = nn.Sequential(
            nn.Linear(pe_emb_dim + gnn_emb_dim, final_emb_dim * 4),
            nn.Tanh(),
            nn.Linear(final_emb_dim * 4, final_emb_dim * 2),
            nn.Tanh(),
            nn.Linear(final_emb_dim * 2, final_emb_dim)
        )

        # Monotonic constraints setup
        if KNN:
            in_dim = final_emb_dim + 2  # includes tau and ybar
            monotonic_constraints = [0] * final_emb_dim + [1, 0]
        else:
            in_dim = final_emb_dim + 1  # only includes tau
            monotonic_constraints = [0] * final_emb_dim + [1]

        net = nn.Sequential(
            lmn.LipschitzLinear(in_dim, 32, kind="one-inf"),
            lmn.GroupSort(2),
            lmn.LipschitzLinear(32, num_features_out, kind="inf")
        )

        # Monotonic network for quantile regression
        self.monotonic_subnet = lmn.MonotonicWrapper(
            lipschitz_module=net,
            lipschitz_const=1.0,
            monotonic_constraints=monotonic_constraints
        )

        # Optional auxiliary task for Moran's I
        if MAT:
            self.fc_morans = lmn.LipschitzLinear(
                final_emb_dim, num_features_out, kind="inf"
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ei: torch.Tensor | None,
        ew: torch.Tensor | None,
        tau: torch.Tensor,
        ybar: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the PEGQAT model.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape [num_nodes, num_features_in].
        c : torch.Tensor
            Node coordinates, shape [num_nodes, coord_dim].
        ei : torch.Tensor | None
            Edge indices if precomputed. If None, KNN will be constructed on the fly.
        ew : torch.Tensor | None
            Edge weights if precomputed. If None, they will be computed on the fly.
        tau : torch.Tensor
            Quantile levels for each node, shape [num_nodes].
        ybar : torch.Tensor
            Optional additional feature (e.g., local average or uncertainty),
            shape [num_nodes].

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If MAT is False, returns the quantile regression output. If MAT is True,
            returns (quantile output, Moran's I output).
        """
        x = x.float().to(self.device)
        c = c.float().to(self.device)
        tau = tau.float().to(self.device)
        ybar = ybar.float().to(self.device)

        # Handle graph edges
        if torch.is_tensor(ei) and torch.is_tensor(ew):
            edge_index = ei
            edge_weight = ew
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            edge_weight = makeEdgeWeight(c, edge_index).to(self.device)

        # GAT forward pass
        x_emb = F.relu(self.conv1(x, edge_index, edge_weight))
        x_emb = F.dropout(x_emb, self.p_dropout, training=self.training)
        x_emb = F.relu(self.conv2(x_emb, edge_index, edge_weight))
        x_emb = F.dropout(x_emb, self.p_dropout, training=self.training)

        # Positional encoder forward pass
        c_reshaped = c.reshape(1, c.shape[0], c.shape[1])  # shape [1, num_nodes, coord_dim]
        c_emb = self.spenc(c_reshaped.detach().cpu().numpy())  # shape [1, num_nodes, pe_hidden_dim]
        c_emb = c_emb.reshape(c_emb.shape[1], c_emb.shape[2])
        c_emb = self.dec_pe(c_emb).float().to(self.device)

        # Merge GAT and positional embeddings
        l_emb = torch.cat((c_emb, x_emb), dim=1)
        phi_emb = self.dec(l_emb).float()

        # Build monotonic input
        tau = tau.view(-1, 1)
        phi_til_emb = torch.cat((phi_emb, tau), dim=1)
        if self.KNN:
            ybar = ybar.view(-1, 1)
            phi_til_emb = torch.cat((phi_til_emb, ybar), dim=1)

        # Monotonic regression output (quantile)
        output = self.monotonic_subnet(phi_til_emb)

        # Auxiliary task (Moran's I) if enabled
        if self.MAT:
            morans_output = self.fc_morans(phi_emb)
            return output, morans_output
        return output


class PEGQSAGE(nn.Module):
    """
    GraphSAGE with a positional encoder, quantile regression (as proposed by Si, 2020),
    and optional ybar input.
    """

    def __init__(
        self,
        num_features_in: int = 6,
        num_features_out: int = 1,
        gnn_hidden_dim: int = 32,
        gnn_emb_dim: int = 32,
        pe_hidden_dim: int = 128,
        pe_emb_dim: int = 64,
        final_emb_dim: int = 8,
        k: int = 5,
        p_dropout: float = 0.5,
        MAT: bool = False,
        KNN: bool = True
    ) -> None:
        """
        Initialize the PEGQSAGE model.

        Parameters
        ----------
        num_features_in : int, optional
            Number of input features for the GCN, by default 6.
        num_features_out : int, optional
            Number of output features (e.g., for quantile regression), by default 1.
        gnn_hidden_dim : int, optional
            Dimension of the GCN hidden layer, by default 32.
        gnn_emb_dim : int, optional
            Dimension of the GCN embedding layer, by default 32.
        pe_hidden_dim : int, optional
            Dimension of the spatial encoder hidden layer, by default 128.
        pe_emb_dim : int, optional
            Dimension of the spatial encoder embedding layer, by default 64.
        final_emb_dim : int, optional
            Dimension of the final merged embedding, by default 8.
        k : int, optional
            Number of nearest neighbors for the KNN graph, by default 5.
        p_dropout : float, optional
            Dropout probability, by default 0.5.
        MAT : bool, optional
            If True, enable an auxiliary task for Moran's I, by default False.
        KNN : bool, optional
            If True, include `ybar` as an additional input to the monotonic subnet.
            By default True.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_emb_dim = gnn_emb_dim
        self.pe_hidden_dim = pe_hidden_dim
        self.pe_emb_dim = pe_emb_dim
        self.final_emb_dim = final_emb_dim
        self.k = k
        self.p_dropout = p_dropout
        self.MAT = MAT
        self.KNN = KNN

        # GraphSAGE layers
        self.conv1 = SAGEConv(num_features_in, gnn_hidden_dim)
        self.conv2 = SAGEConv(gnn_hidden_dim, gnn_emb_dim)

        # Spatial encoder
        self.spenc = GridCellSpatialRelationEncoder(
            spa_embed_dim=pe_hidden_dim,
            ffn=True,
            min_radius=1e-6,
            max_radius=360
        )
        self.dec_pe = nn.Sequential(
            nn.Linear(pe_hidden_dim, pe_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 2, pe_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(pe_hidden_dim // 4, pe_emb_dim)
        )

        # Merge GraphSAGE and positional embeddings
        self.dec = nn.Sequential(
            nn.Linear(pe_emb_dim + gnn_emb_dim, final_emb_dim * 4),
            nn.Tanh(),
            nn.Linear(final_emb_dim * 4, final_emb_dim * 2),
            nn.Tanh(),
            nn.Linear(final_emb_dim * 2, final_emb_dim)
        )

        # Monotonic constraints setup
        if KNN:
            in_dim = final_emb_dim + 2  # includes tau and ybar
            monotonic_constraints = [0] * final_emb_dim + [1, 0]
        else:
            in_dim = final_emb_dim + 1  # only includes tau
            monotonic_constraints = [0] * final_emb_dim + [1]

        net = nn.Sequential(
            lmn.LipschitzLinear(in_dim, 32, kind="one-inf"),
            lmn.GroupSort(2),
            lmn.LipschitzLinear(32, num_features_out, kind="inf")
        )

        # Monotonic network for quantile regression
        self.monotonic_subnet = lmn.MonotonicWrapper(
            lipschitz_module=net,
            lipschitz_const=1.0,
            monotonic_constraints=monotonic_constraints
        )

        # Optional auxiliary task for Moran's I
        if MAT:
            self.fc_morans = lmn.LipschitzLinear(
                final_emb_dim, num_features_out, kind="inf"
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ei: torch.Tensor | None,
        ew: torch.Tensor | None,
        tau: torch.Tensor,
        ybar: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the PEGQSAGE model.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape [num_nodes, num_features_in].
        c : torch.Tensor
            Node coordinates, shape [num_nodes, coord_dim].
        ei : torch.Tensor | None
            Edge indices if precomputed. If None, KNN will be constructed on the fly.
        ew : torch.Tensor | None
            Edge weights if precomputed. If None, they will be computed on the fly.
        tau : torch.Tensor
            Quantile levels for each node, shape [num_nodes].
        ybar : torch.Tensor
            Optional additional feature (e.g., local average or uncertainty),
            shape [num_nodes].

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If MAT is False, returns the quantile regression output. If MAT is True,
            returns (quantile output, Moran's I output).
        """
        x = x.float().to(self.device)
        c = c.float().to(self.device)
        tau = tau.float().to(self.device)
        ybar = ybar.float().to(self.device)

        # Handle graph edges
        if torch.is_tensor(ei) and torch.is_tensor(ew):
            edge_index = ei
            # Edge weights are not directly used in SAGEConv here
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            _ = makeEdgeWeight(c, edge_index).to(self.device)  # not used directly

        # GraphSAGE forward pass
        x_emb = F.relu(self.conv1(x, edge_index))
        x_emb = F.dropout(x_emb, self.p_dropout, training=self.training)
        x_emb = F.relu(self.conv2(x_emb, edge_index))
        x_emb = F.dropout(x_emb, self.p_dropout, training=self.training)

        # Positional encoder forward pass
        c_reshaped = c.reshape(1, c.shape[0], c.shape[1])  # shape [1, num_nodes, coord_dim]
        c_emb = self.spenc(c_reshaped.detach().cpu().numpy())  # shape [1, num_nodes, pe_hidden_dim]
        c_emb = c_emb.reshape(c_emb.shape[1], c_emb.shape[2])
        c_emb = self.dec_pe(c_emb).float().to(self.device)

        # Merge GraphSAGE and positional embeddings
        l_emb = torch.cat((c_emb, x_emb), dim=1)
        phi_emb = self.dec(l_emb).float()

        # Build monotonic input
        tau = tau.view(-1, 1)
        phi_til_emb = torch.cat((phi_emb, tau), dim=1)
        if self.KNN:
            ybar = ybar.view(-1, 1)
            phi_til_emb = torch.cat((phi_til_emb, ybar), dim=1)

        # Monotonic regression output (quantile)
        output = self.monotonic_subnet(phi_til_emb)

        # Auxiliary task (Moran's I) if enabled
        if self.MAT:
            morans_output = self.fc_morans(phi_emb)
            return output, morans_output
        return output


class LossWrapperQuantile(nn.Module):
    """
    A wrapper that computes quantile loss (pinball loss) for a single-task
    quantile regression GNN model.
    """

    def __init__(
        self,
        model: nn.Module,
        task_num: int = 1,
        uw: bool = False,
        lamb: float = 0.0,
        k: int = 5,
        batch_size: int = 2048
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.task_num = task_num
        self.uw = uw
        self.lamb = lamb
        self.k = k
        self.batch_size = batch_size

        # For multi-task settings (not used in this wrapper)
        if self.task_num > 1:
            self.log_vars = nn.Parameter(torch.zeros(self.task_num))

    def forward(
        self,
        input_data: torch.Tensor,
        targets: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
        morans_input: torch.Tensor | None,
        tau: torch.Tensor,
        ybar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the quantile regression loss (pinball loss) for a single-task scenario.

        Parameters
        ----------
        input_data : torch.Tensor
            Node features.
        targets : torch.Tensor
            Ground-truth values.
        coords : torch.Tensor
            Node coordinates.
        edge_index : torch.Tensor | None
            Precomputed edge indices (if any).
        edge_weight : torch.Tensor | None
            Precomputed edge weights (if any).
        morans_input : torch.Tensor | None
            Unused in single-task mode.
        tau : torch.Tensor
            Quantile levels for each sample.
        ybar : torch.Tensor
            An auxiliary feature (e.g., local average) for the model, if used.

        Returns
        -------
        torch.Tensor
            The mean pinball loss.
        """
        if self.task_num == 1:
            outputs = self.model(
                input_data,
                coords,
                edge_index,
                edge_weight,
                probit(tau),
                ybar
            )
            return self.pinball_loss(
                targets.float().view(-1),
                outputs.float().view(-1),
                tau.float().view(-1)
            )
        else:
            raise ValueError("PEGQNN can only be used with task_num=1.")

    def pinball_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Pinball loss for quantile regression.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground-truth values.
        y_pred : torch.Tensor
            Model predictions.
        tau : torch.Tensor
            Quantile levels.

        Returns
        -------
        torch.Tensor
            The mean pinball loss.
        """
        if y_true.size() != tau.size():
            raise ValueError("The size of y_true and tau must match.")

        delta = y_true - y_pred
        loss = torch.where(delta > 0, tau * delta, (tau - 1.0) * delta)
        return loss.mean()
