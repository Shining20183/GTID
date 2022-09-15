import torch
from torch.utils.data import Dataset, DataLoader
import os
import csv
import pandas as pd
import numpy as np
import math
import torch.nn.functional as F


def hex_to_int(s):
    dict0 = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12,
             'd': 13, 'e': 14, 'f': 15}
    return dict0[s]


def hex_str_to_arr(str):
    if str == 'z':
        return []
    else:
        return list(map(hex_to_int, str))


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    # print(len(batch[0]['feature']))
    print('feature: ', batch[0]['feature'])
    print(len(batch[0]['feature']))
    batch = list(zip(*batch))
    print('batch: ', batch)
    # tmp = torch.tensor(batch[0])
    # print('tmp.size(): ', tmp.size())
    # labels = torch.tensor(batch[0], dtype=torch.int32)
    # texts = batch[1]
    # del batch
    # return labels, texts


class MyDatasetSLForTransDNNT(Dataset):
    def __init__(self, root_dir, sl_sum_dir, time_dir, names_file, pad_size, embed, max_time_position, gran, log_e,
                 transform=None):
        self.root_dir = root_dir
        self.sl_sum_dir = sl_sum_dir
        self.time_dir = time_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.name_list = []
        self.pad_size = pad_size
        self.embed = embed
        self.max_time_position = max_time_position
        self.gran = gran
        self.log_e = log_e

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        f = open(self.names_file, 'r')
        reader = csv.reader(f)
        for line in reader:
            self.name_list.append(line)
            self.size += 1

        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / self.embed)) for i in range(self.embed)] for pos in
             range(self.max_time_position)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # 偶数列用sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # 奇数列用cos

    def __len__(self):
        return self.size

    def get_time(self, time_position):
        # 根据时间位置切分出对应的位置编码
        pe = torch.index_select(self.pe, 0, time_position)
        return pe

    def __getitem__(self, idx):
        item = self.name_list[idx]
        feature_csv_path = os.path.join(self.root_dir, item[0])
        label = eval(item[1])
        if not os.path.exists(feature_csv_path):
            print(feature_csv_path, ' does not exist!')
            return None
        feature_f = open(feature_csv_path, 'r')
        feature_reader = csv.reader(feature_f)

        ip_header = feature_reader.__next__()[1:]
        ip_header = np.array(list(map(hex_str_to_arr, list(ip_header))))[:self.pad_size]
        tcp_header = feature_reader.__next__()[1:]
        tcp_header = np.array(list(map(hex_str_to_arr, list(tcp_header))))[:self.pad_size]
        header = np.hstack((ip_header, tcp_header))
        header = torch.from_numpy(header)

        slsum_csv_path = os.path.join(self.sl_sum_dir, item[0])
        if not os.path.exists(slsum_csv_path):
            print(slsum_csv_path, 'does not exist!')
            return None
        sl_sum = pd.read_csv(slsum_csv_path, header=None, index_col=None).values[:self.pad_size]
        sl_sum = torch.from_numpy(np.array(sl_sum))

        ori_seq_len = header.shape[0]
        pad_len = self.pad_size - ori_seq_len

        header = F.pad(header.T, (0, pad_len)).T.numpy()
        sl_sum = F.pad(sl_sum.T, (0, pad_len)).T.numpy()

        if pad_len == 0:
            mask = np.array([False] * ori_seq_len)
        else:
            mask = np.concatenate((np.array([False] * ori_seq_len), np.array([True] * pad_len)))  # padding mask

        # time
        time_csv_path = os.path.join(self.time_dir, item[0])
        time_record = pd.read_csv(time_csv_path, header=None, index_col=None).values[0][:self.pad_size]
        len_time_record = len(time_record)
        for i in range(len_time_record):
            value = round(math.log(round(time_record[i] / self.gran) + 1, self.log_e))
            time_record[i] = value
        for j in range(self.pad_size - len_time_record):
            time_record = np.append(time_record, time_record[len_time_record - 1])

        time_feature = self.get_time(torch.IntTensor(time_record))

        sample = {'header': header, 'sl_sum': sl_sum, 'mask': mask, 'time': time_feature, 'label': label, 'idx': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample

