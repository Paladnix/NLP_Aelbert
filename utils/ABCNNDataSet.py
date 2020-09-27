import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import os
import pandas as pd
from transformers import BertTokenizer


import logging

current_path = os.path.dirname(__file__)
DATA_DIR = current_path + "/../data/qqp"
TRAIN_DATA = DATA_DIR + "/qqp_train.csv"
TEST_DATA = DATA_DIR + "/qqp_dev.csv"

MAX_LEN = 64
PAD_ID = 0
SEP_ID = 102

class ABCNNDataset(Dataset):

    def __init__(self, datafile, sep=','):
        self.datafile = datafile
        self.df = pd.read_csv(self.datafile, sep=sep)
        self.len = self.df.shape[0]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        logging.info("数据集大小: %d", self.len)


    def __len__(self):
        return self.len

    def __getitem__(self, i):
        row = self.df.iloc[i % self.len]
        ids_1, ids_2 = self.get_token_ids(row)
        return ids_1, ids_2, torch.tensor(row.label), str(row.sentence1)+"\n"+str(row.sentence2)

    def pad_slice(self, tokens, length):
        if len(tokens) <= length:
            return tokens
        else:
            return tokens[:length]

    def get_seg_ids(self, ids):
        seg_ids = []
        tag = 0
        for x in ids:
            seg_ids += [tag]
            if x == SEP_ID:
                tag += 1
        return seg_ids 

    def get_token_ids(self, row):
        if row.sentence1 is None or row.sentence2 is None:
            return None, None
        if type(row.sentence1) != type("a") or type(row.sentence2) != type('a'):
            return None, None
        if len(row.sentence1) == 0 or len(row.sentence2) == 0:
            return None, None
        ids_1 = self.tokenizer.encode(row.sentence1, max_length=MAX_LEN, add_special_tokens=True)
        ids_2 = self.tokenizer.encode(row.sentence2, max_length=MAX_LEN, add_special_tokens=True)
        #padding
        if len(ids_1) < MAX_LEN:
            ids_1 += [PAD_ID] * (MAX_LEN- len(ids_1))
        if len(ids_2) < MAX_LEN:
            ids_2 += [PAD_ID] * (MAX_LEN- len(ids_2))
        #totensor
        ids_1 = torch.tensor(ids_1)[:MAX_LEN]
        ids_2 = torch.tensor(ids_2)[:MAX_LEN]

        return ids_1, ids_2

def my_collate_fn(batch):
    # 过滤为None的数据
    batch = list(filter(lambda x:x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据


def get_loader(df, batch_size=16, is_train=True, sep=','):
    ds_df = ABCNNDataset(df, sep)
    loader = torch.utils.data.DataLoader(ds_df, collate_fn=my_collate_fn, batch_size=batch_size, shuffle=is_train, num_workers=4, drop_last=is_train)
    loader.num = len(ds_df)
    return loader


def get_train_loader():
    return get_loader(TRAIN_DATA)

def get_test_loader():
    return get_loader(TEST_DATA, is_train=False)



if __name__ == '__main__':
    df = pd.read_csv(TRAIN_DATA)
    yes = df.label.value_counts("1")
    print(yes)
