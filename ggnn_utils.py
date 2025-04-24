import datetime
import numpy as np
import torch
from torch_geometric.data import Batch, Data


class SimpleBatch(list):

    def to(self, device):
        for ele in self:
            ele.to(device)
        return self


def collate_wrapper(x, y, edge_index, edge_attr, device, return_y=True):
    x = torch.tensor(x, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.float, device=device)
    x = x.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    y_T_first = y.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    #  do not tranpose y_truth
    T = x.size()[0]
    N = x.size()[1]

    # generate batched sequence.
    sequences = []
    for t in range(T):
        cur_batch_x = x[t]
        cur_batch_y = y_T_first[t]
        batch = Batch.from_data_list([
            Data(x=cur_batch_x[i],
                 edge_index=edge_index,
                 edge_attr=edge_attr,
                 y=cur_batch_y[i]) for i in range(N)
        ])
        sequences.append(batch)
    if return_y:
        return SimpleBatch(sequences), y
    else:
        return SimpleBatch(sequences)


def collate_wrapper_multi_branches(x_numpy, y_numpy, edge_index_list, device):
    sequences_multi_branches = []
    for edge_index in edge_index_list:
        sequences, y = collate_wrapper(x_numpy, y_numpy, edge_index, device, return_y=True)
        sequences_multi_branches.append(sequences)

    return sequences_multi_branches, y



def append_time_info(x, y, xtime, ytime, device):
    # 将 xtime 转换为 weekofday (0=Monday, 6=Sunday) from timestamp 不使用utc timezone
    xtime = xtime.astype("datetime64[ns]").astype(int) / 1e9
    xtime_datetime = np.vectorize(lambda x: datetime.datetime.fromtimestamp(x))(xtime)
    
    # dow and tod append to the raw feature.
    x_dow = np.vectorize(lambda x: x.weekday())(xtime_datetime)
    x_tod = np.vectorize(lambda x: (x.hour * 60 + x.minute) / 15 % 96 / 96)(xtime_datetime) # 15 min interval
    B, T, N, F = x.shape
    x_dow = np.expand_dims(x_dow, axis=(-1, -2)).repeat(N, axis=-2)
    x_tod = np.expand_dims(x_tod, axis=(-1, -2)).repeat(N, axis=-2)
    x = np.concatenate((x, x_tod, x_dow), axis=-1)

    # Apply to label data in case used in the downstream task.
    ytime = ytime.astype("datetime64[ns]").astype(int) / 1e9
    ytime_datetime = np.vectorize(lambda x: datetime.datetime.fromtimestamp(x))(ytime)
    y_dow = np.vectorize(lambda x: x.weekday())(ytime_datetime)
    y_tod = np.vectorize(lambda x: (x.hour * 60 + x.minute) / 15 % 96 / 96)(ytime_datetime)
    B, T, N, F = x.shape
    y_dow = np.expand_dims(y_dow, axis=(-1, -2)).repeat(N, axis=-2)
    y_tod = np.expand_dims(y_tod, axis=(-1, -2)).repeat(N, axis=-2)
    y = np.concatenate((y, y_tod, y_dow), axis=-1)
    x = torch.tensor(x, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.float, device=device)
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("Input (x) contains NaN or Inf!")
    if torch.isnan(y).any() or torch.isinf(y).any():
        print("Input contains NaN or Inf!")
    return x, y


def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')