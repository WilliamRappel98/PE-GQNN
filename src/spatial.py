from math import radians, cos, sin, asin, sqrt, pow
import numpy as np
from scipy.stats import wasserstein_distance
import torch


def haversine(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float
) -> float:
    """
    Calculate the great circle distance between two points on the Earth
    (specified in decimal degrees).

    Parameters
    ----------
    lon1 : float
        Longitude of the first point.
    lat1 : float
        Latitude of the first point.
    lon2 : float
        Longitude of the second point.
    lat2 : float
        Latitude of the second point.

    Returns
    -------
    float
        Distance between the two points in meters.
    """
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r * 1000  # convert to meters


def newDistance(
    a: torch.Tensor | np.ndarray,
    b: torch.Tensor | np.ndarray,
    nd_dist: str = "great_circle"
) -> float | torch.Tensor:
    """
    Compute the distance between two points in 2D, 3D, or higher dimensions.

    Parameters
    ----------
    a : torch.Tensor | np.ndarray
        Coordinates of the first point. May have dimension 2, 3, or more.
    b : torch.Tensor | np.ndarray
        Coordinates of the second point. Same dimensionality as `a`.
    nd_dist : str, optional
        Specifies the distance metric. Options are ["great_circle", "euclidean",
        "wasserstein"]. For 2D, defaults to 'great_circle'. For 3D or more,
        defaults to Euclidean unless 'wasserstein' is specified. Defaults to "great_circle".

    Returns
    -------
    float | torch.Tensor
        The computed distance. If dimensionality is >3 and 'wasserstein' is not
        used, a torch.Tensor of squared distances is returned.
    """
    if a.shape[0] == 2:
        x1, y1 = a[0], a[1]
        x2, y2 = b[0], b[1]
        if nd_dist == "euclidean":
            d = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        else:  # great_circle distance
            d = haversine(x1, y1, x2, y2)
    elif a.shape[0] == 3:
        x1, y1, z1 = a[0], a[1], a[2]
        x2, y2, z2 = b[0], b[1], b[2]
        d = sqrt(
            pow(x2 - x1, 2)
            + pow(y2 - y1, 2)
            + pow(z2 - z1, 2) * 1.0
        )
    else:
        # For higher dimensions
        if nd_dist == "wasserstein":
            # Detach in case they are torch Tensors with gradients
            a_np = a.reshape(-1).detach()
            b_np = b.reshape(-1).detach()
            d = wasserstein_distance(a_np, b_np)
        else:
            # Calculate squared distance in a broadcasted manner
            d = torch.pow(
                a.reshape(1, 1, -1) - b.reshape(1, 1, -1),
                2
            ).sum(dim=2)
    return d


def makeEdgeWeight(
    c: torch.Tensor | np.ndarray,
    edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Construct edge weights for a graph based on distances between node coordinates.

    Parameters
    ----------
    c : torch.Tensor | np.ndarray
        Node coordinates, shape [num_nodes, dimension].
    edge_index : torch.Tensor
        Tensor of shape [2, num_edges], each column is (to, from).

    Returns
    -------
    torch.Tensor
        A 1D tensor of edge weights, scaled into the [0,1] range,
        where 1 indicates the closest nodes.
    """
    to_idx = edge_index[0]
    from_idx = edge_index[1]

    # Calculate distances for each edge
    edge_dists = []
    for i in range(len(to_idx)):
        edge_dists.append(newDistance(c[to_idx[i]], c[from_idx[i]]))

    # Scale distances to edge weights in [0,1] range
    max_dist = max(edge_dists)
    min_dist = min(edge_dists)
    scaled_weights = [
        (max_dist - dist) / (max_dist - min_dist) for dist in edge_dists
    ]

    return torch.tensor(scaled_weights, dtype=torch.float)


def knn_to_adj(
    knn: tuple[torch.Tensor, torch.Tensor],
    n: int
) -> torch.Tensor:
    """
    Convert a KNN graph (specified by edge indices) into an adjacency matrix.

    Parameters
    ----------
    knn : tuple[torch.Tensor, torch.Tensor]
        A tuple of (to_indices, from_indices) for edges.
    n : int
        Total number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        An n x n adjacency matrix.
    """
    adj_matrix = torch.zeros(n, n, dtype=torch.float)
    to_indices, from_indices = knn
    for i in range(len(to_indices)):
        to_node = to_indices[i]
        from_node = from_indices[i]
        adj_matrix[to_node, from_node] = 1.0
    # Return the transpose for consistency with the typical (row -> column) usage
    return adj_matrix.T


def normal_torch(
    tensor: torch.Tensor,
    min_val: int = 0
) -> torch.Tensor:
    """
    Normalize a torch.Tensor to [0,1] or [-1,1] range.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be normalized.
    min_val : int, optional
        If 0, normalize to [0,1]. If -1, normalize to [-1,1].
        Defaults to 0.

    Returns
    -------
    torch.Tensor
        A copy of the normalized tensor with gradients detached.
    """
    t_min = torch.min(tensor)
    t_max = torch.max(tensor)
    # Handle the all-zero case
    if t_min == 0 and t_max == 0:
        return tensor.clone().detach().requires_grad_(True)

    if min_val == -1:
        tensor_norm = 2 * ((tensor - t_min) / (t_max - t_min)) - 1
    else:  # min_val == 0
        tensor_norm = (tensor - t_min) / (t_max - t_min)

    return tensor_norm.clone().detach().requires_grad_(True)


def lw_tensor_local_moran(
    y: torch.Tensor,
    w_sparse: np.ndarray | torch.Tensor,
    na_to_zero: bool = True,
    norm: bool = True,
    norm_min_val: int = 0
) -> torch.Tensor:
    """
    Compute the local Moran's I for each element in a 1D torch.Tensor.

    Parameters
    ----------
    y : torch.Tensor
        Values (e.g., a spatial variable), shape [n].
    w_sparse : np.ndarray | torch.Tensor
        Spatial weight matrix in sparse or dense form, same size as `y`.
    na_to_zero : bool, optional
        If True, NaN values in the result are replaced with zero. Defaults to True.
    norm : bool, optional
        If True, normalize the result using `normal_torch`. Defaults to True.
    norm_min_val : int, optional
        If 0, normalizes to [0,1], if -1, normalizes to [-1,1].
        Defaults to 0.

    Returns
    -------
    torch.Tensor
        A 1D tensor of local Moran's I values, optionally normalized.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_flat = y.reshape(-1)
    n = len(y_flat)
    n_1 = n - 1

    # Center and scale
    z = y_flat - y_flat.mean()
    sy = y_flat.std()
    z /= sy

    # Denominator
    den = (z * z).sum()

    # Multiply by the spatial weights
    z_detach = z.detach()
    zl = torch.tensor(w_sparse * z_detach).to(device)

    # Compute local Moran's I
    mi = n_1 * z * zl / den

    # Handle NaNs
    if na_to_zero:
        mi[torch.isnan(mi)] = 0

    # Optional normalization
    if norm:
        mi = normal_torch(mi, min_val=norm_min_val)

    return mi.clone().detach().requires_grad_(True)
