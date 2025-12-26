from data import *
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
from torch.utils.data import Dataset
import torch.utils.data
import torch.utils.data.distributed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_load(dataset, train_size, val_size, random_state, train_sub_sample, eval="test", x=None, y=None, coords=None):
    if dataset == "california_housing":
        x, y, c = get_california_housing_data(norm_x=False)
    elif dataset == "air_temp":
        x, y, c = get_air_temp_data(norm_x=False)
    elif dataset == "3d_road":
        x, y, c = get_3d_road_data()
        x = torch.randn(y.shape[0], 1)
    elif dataset == "australia":
        x, y, c = get_australia_data(aux=(x, y, coords))

    # Split data
    n = x.shape[0]
    indices = np.arange(n)
    _, _, _, _, idx_train, idx_val_test = train_test_split(
        x, y, indices, test_size=(1 - train_size), random_state=random_state
    )
    idx_val, idx_test = train_test_split(
        idx_val_test,
        test_size=(1 - train_size - val_size) / (1 - train_size),
        random_state=random_state,
    )

    # Perform subsample on idx_train
    _, idx_train = train_test_split(
        idx_train,
        test_size=train_sub_sample / train_size,
        random_state=random_state,
    )

    # Separate c, x, and y objects
    train_c, val_c, test_c = c[idx_train], c[idx_val], c[idx_test]
    train_x, val_x, test_x = x[idx_train], x[idx_val], x[idx_test]
    train_y, val_y, test_y = y[idx_train], y[idx_val], y[idx_test]

    # Convert tensors to numpy arrays
    train_c_np, val_c_np, test_c_np = train_c.numpy(), val_c.numpy(), test_c.numpy()
    train_x_np, val_x_np, test_x_np = train_x.numpy(), val_x.numpy(), test_x.numpy()
    train_y_np, val_y_np, test_y_np = train_y.numpy(), val_y.numpy(), test_y.numpy()

    # Add an extra dimension
    if train_y_np.ndim == 1:
        train_y_np = np.expand_dims(train_y_np, axis=1)
        val_y_np = np.expand_dims(val_y_np, axis=1)
        test_y_np = np.expand_dims(test_y_np, axis=1)
    if train_x_np.ndim == 1:
        train_x_np = np.expand_dims(train_x_np, axis=1)
        val_x_np = np.expand_dims(val_x_np, axis=1)
        test_x_np = np.expand_dims(test_x_np, axis=1)

    # Concatenate numpy arrays: first columns from c, then from x, then from y
    train_data_np = np.concatenate((train_c_np, train_x_np, train_y_np), axis=1)
    val_data_np = np.concatenate((val_c_np, val_x_np, val_y_np), axis=1)
    test_data_np = np.concatenate((test_c_np, test_x_np, test_y_np), axis=1)

    # Create column names (adjust according to the number of columns in c, x, y)
    c_columns = [f"c{i}" for i in range(train_c_np.shape[1])]
    x_columns = [f"x{i}" for i in range(train_x_np.shape[1])]
    y_columns = [f"y{i}" for i in range(train_y_np.shape[1])]

    # Combine all column names
    columns = c_columns + x_columns + y_columns

    # Create the DataFrames
    train_data = pd.DataFrame(train_data_np, columns=columns).reset_index()
    val_data = pd.DataFrame(val_data_np, columns=columns).reset_index()
    test_data = pd.DataFrame(test_data_np, columns=columns).reset_index()

    train_data = np.array(train_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)
    train_x = train_data[:, 1:-1]
    train_y = train_data[:, [0, -1]]
    val_x = val_data[:, 1:-1]
    val_y = val_data[:, [0, -1]]
    test_x = test_data[:, 1:-1]
    test_y = test_data[:, [0, -1]]

    # standardization
    scaler = StandardScaler()  # standard
    x = torch.FloatTensor(scaler.fit_transform(train_x)).to(device).float()
    y = torch.tensor(train_y).to(device).float()
    val_x = torch.FloatTensor(scaler.transform(val_x)).to(device).float()
    val_y = torch.tensor(val_y).to(device).float().squeeze(-1)
    test_x = torch.FloatTensor(scaler.transform(test_x)).to(device).float()
    test_y = torch.tensor(test_y).to(device).float().squeeze(-1)
    
    # return
    if eval == "test":
        return x, y, test_x, test_y
    elif eval == "validation":
        return x, y, val_x, val_y


def get_tensor_from_pd(dataframe_series):
    return torch.tensor(data=dataframe_series.values)


class DatasetGP(Dataset):
    def __init__(
        self,
        n_tasks,
        batch_size=2,
        n_context_min=3,
        n_context_max=600,
        dataset="california_housing",
        train_size=0.8,
        val_size=0.1,
        random_state=42,
        train_sub_sample=0.1,
        eval="test",
        x=None,
        y=None,
        coords=None,
    ):

        super().__init__()
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.n_context_min = n_context_min
        self.n_context_max = n_context_max
        self.dataset = dataset
        self.train_size = train_size
        self.val_size = val_size
        self.random_state = random_state
        self.train_sub_sample = train_sub_sample
        self.eval = eval
        self.x = x
        self.y = y
        self.coords = coords

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, index):
        n_context = np.random.randint(self.n_context_min, self.n_context_max + 1)
        n_target = n_context + np.random.randint(3, 750 - n_context + 1)

        batch_context_x = []
        batch_context_y = []
        batch_target_x = []
        batch_target_y = []

        for _ in range(self.batch_size):
            (
                x,
                y,
                _,
                _,
            ) = data_load(
                self.dataset,
                self.train_size,
                self.val_size,
                self.random_state,
                self.train_sub_sample,
                self.eval,
                self.x,
                self.y,
                self.coords,
            )
            context_x = x[0:n_context, :]
            context_y = y[0:n_context, :]

            target_x = x[0:n_target, :]
            target_y = y[0:n_target, :]

            batch_context_x.append(context_x)
            batch_context_y.append(context_y)

            batch_target_x.append(target_x)
            batch_target_y.append(target_y)

        batch_context_x = torch.stack(batch_context_x, dim=0)
        batch_context_y = torch.stack(batch_context_y, dim=0)
        batch_target_x = torch.stack(batch_target_x, dim=0)
        batch_target_y = torch.stack(batch_target_y, dim=0)

        return batch_context_x, batch_context_y, batch_target_x, batch_target_y


class DatasetGP_test(Dataset):
    def __init__(
        self,
        n_tasks,
        batch_size=1,
        dataset="california_housing",
        train_size=0.8,
        val_size=0.1,
        random_state=42,
        train_sub_sample=0.1,
        eval="test",
        x=None,
        y=None,
        coords=None,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.dataset = dataset
        self.train_size = train_size
        self.val_size = val_size
        self.random_state = random_state
        self.train_sub_sample = train_sub_sample
        self.eval = eval
        self.x = x
        self.y = y
        self.coords = coords

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, index):
        batch_context_x = []
        batch_context_y = []
        batch_target_x = []
        batch_target_y = []

        for _ in range(self.batch_size):
            x, y, eval_x, eval_y = data_load(
                self.dataset,
                self.train_size,
                self.val_size,
                self.random_state,
                self.train_sub_sample,
                self.eval,
                self.x,
                self.y,
                self.coords,
            )
            context_x = x
            context_y = y

            target_x = eval_x
            target_y = eval_y

            batch_context_x.append(context_x)
            batch_context_y.append(context_y)

            batch_target_x.append(target_x)
            batch_target_y.append(target_y)

        batch_context_x = torch.stack(batch_context_x, dim=0)
        batch_context_y = torch.stack(batch_context_y, dim=0)
        batch_target_x = torch.stack(batch_target_x, dim=0)
        batch_target_y = torch.stack(batch_target_y, dim=0)

        return batch_context_x, batch_context_y, batch_target_x, batch_target_y
