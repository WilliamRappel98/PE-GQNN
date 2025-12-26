import io
import os
import random
from urllib import request

import numpy as np
import pandas as pd
import requests
import sklearn.datasets
import torch
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed to set for all relevant libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def normalize(x: np.ndarray, min_val: int = 0) -> np.ndarray:
    """
    Normalize a vector.

    Parameters
    ----------
    x : np.ndarray
        Numerical vector to be normalized.
    min_val : int, optional
        Choice of [0, -1], setting whether normalization is in the range [0, 1]
        or [-1, 1]. Defaults to 0.

    Returns
    -------
    np.ndarray
        Normalized vector.
    """
    x_min = np.min(x)
    x_max = np.max(x)

    # If the vector is all zeros, just return it as is.
    if x_min == 0 and x_max == 0:
        return x

    if min_val == -1:
        x_norm = 2 * ((x - x_min) / (x_max - x_min)) - 1
    else:  # min_val == 0
        x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def get_california_housing_data(
    norm_x: bool = True,
    norm_y: bool = True,
    min_val: int = 0,
    spat_int: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Download and process the California Housing Dataset.

    Parameters
    ----------
    norm_x : bool, optional
        Whether features should be normalized. Defaults to True.
    norm_y : bool, optional
        Whether the outcome should be normalized. Defaults to True.
    min_val : int, optional
        Choice of [0, -1], sets the normalization range to [0, 1] or [-1, 1].
        Defaults to 0.
    spat_int : bool, optional
        If True, replaces features with a tensor of ones (e.g., intercept only).
        Defaults to False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (x, y, coords), where x and y are features and outcome tensors,
        and coords are the spatial coordinates (latitude and longitude).
    """
    cali_housing_ds = sklearn.datasets.fetch_california_housing()
    x = np.array(cali_housing_ds.data[:, :6])
    y = np.array(cali_housing_ds.target)
    coords = np.array(cali_housing_ds.data[:, 6:])

    if norm_x:
        for i in range(x.shape[1]):
            x[:, i] = normalize(x[:, i], min_val)

    if norm_y:
        y = normalize(y, min_val)

    if spat_int:
        x = torch.ones(x.shape[0], 1)

    return torch.tensor(x), torch.tensor(y), torch.tensor(coords)


def get_air_temp_data(
    norm_x: bool = True,
    norm_y: bool = True,
    min_val: int = 0,
    pred: str = "temp",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Download and process the Global Air Temperature dataset.

    Parameters
    ----------
    norm_x : bool, optional
        Whether features should be normalized. Defaults to True.
    norm_y : bool, optional
        Whether the outcome should be normalized. Defaults to True.
    min_val : int, optional
        Choice of [0, -1], sets the normalization range to [0, 1] or [-1, 1].
        Defaults to 0.
    pred : str, optional
        Outcome variable to be returned; choose from ["temp", "prec"].
        Defaults to "temp".

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (x, y, coords), where x and y are features and outcome tensors,
        and coords are the spatial coordinates (longitude and latitude).
    """
    try:
        inc_df = pd.read_parquet("../../data/air_temperature.parquet")
        inc = np.array(inc_df)
    except Exception:
        url = "https://springernature.figshare.com/ndownloader/files/12609182"
        url_open = request.urlopen(url)
        inc_df = pd.read_csv(io.StringIO(url_open.read().decode("utf-8")))
        inc_df.to_parquet("../../data/air_temperature.parquet")
        inc = np.array(inc_df)

    if pred == "temp":
        x = inc[:, 5]
        y = inc[:, 4].reshape(-1)
    else:  # pred == "prec"
        x = inc[:, 4]
        y = inc[:, 5].reshape(-1)

    coords = inc[:, :2]

    if norm_x:
        x = normalize(x, min_val).reshape(-1, 1)

    if norm_y:
        y = normalize(y, min_val)

    return torch.tensor(x), torch.tensor(y), torch.tensor(coords)


def get_3d_road_data(
    norm_y: bool = True, min_val: int = 0
) -> tuple[None, torch.Tensor, torch.Tensor]:
    """
    Download and process the 3D Road Network dataset.

    Parameters
    ----------
    norm_y : bool, optional
        Whether the outcome should be normalized. Defaults to True.
    min_val : int, optional
        Choice of [0, -1], sets the normalization range to [0, 1] or [-1, 1].
        Defaults to 0.

    Returns
    -------
    tuple[None, torch.Tensor, torch.Tensor]
        (None, y, coords), where y is the outcome tensor, and coords are the
        spatial coordinates (x, y). The features are None for this dataset.
    """
    try:
        c_df = pd.read_parquet("../../data/3d_road.parquet")
    except Exception:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt"
        try:
            s = requests.get(url).content
            c_df = pd.read_csv(io.StringIO(s.decode("utf-8")))
        except Exception:
            c_df = pd.read_csv("../data/3D_spatial_network.txt")
        c_df.columns = ["id", "x", "y", "z"]
        c_df.to_parquet("../../data/3d_road.parquet")

    coords = np.array(c_df[["x", "y"]])
    y = np.array(c_df[["z"]]).reshape(-1)

    if norm_y:
        y = normalize(y, min_val)

    return None, torch.tensor(y), torch.tensor(coords)


def get_australia_data(
    norm_x: bool = True,
    norm_y: bool = True,
    min_val: int = 0,
    spat_int: bool = False,
    aux: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-optimized version: avoids ds.to_numpy() to prevent MemoryError.
    Produces exactly the same output as the original function.
    """
    if aux:
        return aux
    else:
        try:
            ds = pd.read_parquet("../../data/australia.parquet")
        except Exception:
            ds = pd.read_csv("../../data/australia.csv")
            ds.dropna(
                subset=["lat", "lon", "Median_tot_prsnl_inc_weekly"],
                inplace=True,
                ignore_index=True,
            )
            ds = ds[ds["Median_tot_prsnl_inc_weekly"].gt(0)].reset_index(drop=True)
            ds.to_parquet("../../data/australia.parquet")

        y = ds.iloc[:, 3].to_numpy(copy=False)
        coords = ds.iloc[:, 1:3].to_numpy(copy=False)

        feature_cols = ds.columns[4:]
        n_rows = len(ds)

        x_tensors = []
        for col in feature_cols:
            col_data = ds[col].to_numpy(copy=False)
            if norm_x:
                col_data = normalize(col_data, min_val)
            x_tensors.append(torch.from_numpy(col_data.astype(np.float32)))

        x = torch.stack(x_tensors, dim=1)

        if norm_y:
            y = normalize(y, min_val)

        y = torch.from_numpy(y.astype(np.float32))
        coords = torch.from_numpy(coords.astype(np.float32))

        if spat_int:
            x = torch.ones((n_rows, 1), dtype=torch.float32)

        return x, y, coords


def get_australia_data_full(
    norm_x: bool = True,
    norm_y: bool = True,
    min_val: int = 0,
    spat_int: bool = False,
    aux: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Download and process the Australia Census Dataset.

    Parameters
    ----------
    norm_x : bool, optional
        Whether features should be normalized. Defaults to True.
    norm_y : bool, optional
        Whether the outcome should be normalized. Defaults to True.
    min_val : int, optional
        Choice of [0, -1], sets the normalization range to [0, 1] or [-1, 1].
        Defaults to 0.
    spat_int : bool, optional
        If True, replaces features with a tensor of ones (e.g., intercept only).
        Defaults to False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (x, y, coords), where x and y are features and outcome tensors,
        and coords are the spatial coordinates (latitude and longitude).
    """
    if aux:
        return aux
    else:
        try:
            australia_ds = pd.read_parquet("../../data/australia_full_cleaned.parquet")
        except Exception:
            australia_ds = pd.read_parquet("../../data/australia_full.parquet")
            australia_ds.dropna(
                subset=["lat", "lon", "Median_tot_prsnl_inc_weekly"],
                inplace=True,
                ignore_index=True,
            )
            australia_ds = australia_ds[
                australia_ds["Median_tot_prsnl_inc_weekly"].gt(0)
            ].reset_index(drop=True)
            australia_ds.to_parquet("../../data/australia_full_cleaned.parquet")
        x = np.array(australia_ds.iloc[:, 4:])
        y = np.array(australia_ds.iloc[:, 3])
        coords = np.array(australia_ds.iloc[:, 1:3])

        if norm_x:
            for i in range(x.shape[1]):
                x[:, i] = normalize(x[:, i], min_val)

        if norm_y:
            y = normalize(y, min_val)

        if spat_int:
            x = torch.ones(x.shape[0], 1)

        return torch.tensor(x), torch.tensor(y), torch.tensor(coords)


class MyDataset(Dataset):
    """
    Custom PyTorch Dataset that stores features, targets, and coordinates,
    and optionally an additional target vector (ybar).
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        coords: torch.Tensor,
        ybar: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        x : torch.Tensor
            The input features.
        y : torch.Tensor
            The target variable.
        coords : torch.Tensor
            The spatial coordinates (longitude, latitude, etc.).
        ybar : torch.Tensor | None, optional
            An additional target variable, if needed. Defaults to None.
        """
        self.features = x
        self.target = y
        self.coords = coords
        self.ybar = ybar

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Retrieve a single sample from the dataset by index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple containing features, target, coordinates, and optionally
            ybar (if it exists).
        """
        if self.ybar is not None:
            return (
                self.features[idx].clone().detach().requires_grad_(True),
                self.target[idx].clone().detach().requires_grad_(True),
                self.coords[idx].clone().detach().requires_grad_(True),
                self.ybar[idx].clone().detach().requires_grad_(True),
            )
        else:
            return (
                self.features[idx].clone().detach().requires_grad_(True),
                self.target[idx].clone().detach().requires_grad_(True),
                self.coords[idx].clone().detach().requires_grad_(True),
            )
