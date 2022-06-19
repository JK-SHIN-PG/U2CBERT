import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd


def get_data_main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    f = open("./data/wiki-0.1percent.txt", "r")
    wiki = f.readlines()
    data = []
    for idx, line in enumerate(wiki):
        result = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(line.strip())[:510])
        # [CLS] : 101, [SEP] : 102, [PAD] : 0
        result = [101] + result + [102] + [0]*(510 - len(result))
        data.append(result)
    t_data = torch.LongTensor(data)
    del data
    return t_data


class KD_datatset(Dataset):
    def __init__(self, data, device):
        self.token_data = data
        self.device = device
        self.batch_size = data.shape[0]
        self.attetion_mask = None
        self.get_attention_mask()

    def get_attention_mask(self):
        mask = torch.ones_like(self.token_data)
        bool_mask = self.token_data != 0  # ..
        atttention_mask = torch.mul(mask, bool_mask)
        self.attetion_mask = atttention_mask

    def __getitem__(self, index):
        return self.token_data[index], self.attetion_mask[index]

    def __len__(self):
        return len(self.token_data)


class FT_dataset(Dataset):
    def __init__(self, data, token_ids, labels, device):
        self.token_data = data
        self.device = device
        self.labels = torch.LongTensor(list(map(int, labels)))
        self.token_ids = token_ids
        self.batch_size = data.shape[0]
        self.attetion_mask = None
        self.get_attention_mask()

    def get_attention_mask(self):
        mask = torch.ones_like(self.token_data)
        bool_mask = self.token_data != 0  # ..
        atttention_mask = torch.mul(mask, bool_mask)
        self.attetion_mask = atttention_mask

    def __getitem__(self, index):
        return self.token_data[index], self.attetion_mask[index], self.token_ids[index], self.labels[index]

    def __len__(self):
        return len(self.token_data)


def load_data(tokenized_data, device, batch_size, shuffle_true=True, ft=False, token_ids=None, labels = None):
    if ft == True:
        dataset = FT_dataset(tokenized_data, token_ids, labels, device)
    else:
        dataset = KD_datatset(tokenized_data, device)
    loader = DataLoader(dataset, shuffle=shuffle_true,
                        batch_size=batch_size)
    print("Complete! Data loading")
    return loader


def get_data(data_type="test", task="stsb"):
    '''
    task list : 'stsb', 'sick', 'mrpc'
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if task == "sick":
        dataset = load_dataset("sick", task)
    else:
        dataset = load_dataset("glue", task)
    if data_type == "train":
        train = np.reshape([data[key] for data in dataset['train'] for key in data], (len(
            dataset['train']), len(dataset['train'].features)))
        df = pd.DataFrame(train, columns=list(
            dataset['train'].features.keys()))
    elif data_type == "valid":
        validation = np.reshape([data[key] for data in dataset['validation'] for key in data], (len(
            dataset['validation']), len(dataset['validation'].features)))
        df = pd.DataFrame(validation, columns=list(
            dataset['validation'].features.keys()))
    else:
        test = np.reshape([data[key] for data in dataset['test'] for key in data], (len(
            dataset['test']), len(dataset['test'].features)))
        df = pd.DataFrame(test, columns=list(
            dataset['test'].features.keys()))

    st_list1 = []
    st_list2 = []
    for i in range(len(df)):
        if task == "sick":
            line1 = df.loc[i, "sentence_A"]
            line2 = df.loc[i, "sentence_B"]
        else:
            line1 = df.loc[i, "sentence1"]
            line2 = df.loc[i, "sentence2"]
        result1 = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(line1.strip())[:510])
        result2 = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(line2.strip())[:510])
        result1 = [101] + result1 + [102] + [0]*(510 - len(result1))
        result2 = [101] + result2 + [102] + [0]*(510 - len(result2))
        st_list1.append(result1)
        st_list2.append(result2)
    t_data1 = torch.LongTensor(st_list1)
    t_data2 = torch.LongTensor(st_list2)
    return t_data1, t_data2, df.label.astype(float).tolist()


def construct_data_for_finetuning(t_data1, t_data2):
    token_ids = torch.zeros_like(t_data1)
    idices = (t_data1 == 102).nonzero(as_tuple=True)[1]
    data = t_data1.clone()
    for i in range(idices.shape[0]):
        data[i][idices[i] + 1:] = t_data2[i, 1:512 - (idices[i])]
        token_ids[i][(idices[i] +1):] = torch.ones_like(t_data2[i, 1:512 - (idices[i])])

    return data, token_ids
