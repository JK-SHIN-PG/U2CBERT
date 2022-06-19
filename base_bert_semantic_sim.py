# %%
from dataload import get_data, load_data
from datasets import load_dataset, load_metric
import pandas as pd
import numpy as np
import torch
from model import studentBertModel, BertConfig
from pytorch_pretrained_bert import BertTokenizer, BertModel
from scipy.stats import spearmanr
import os
from utils import align_loss, uniform_loss
# device setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_num = "0"
device = torch.device(
    f"cuda:{gpu_num}") if torch.cuda.is_available() else torch.device("cpu")

# data setting
t_data1, t_data2, labels = get_data("train")
loader1 = load_data(t_data1, device, 16, shuffle_true=False)
loader2 = load_data(t_data2, device, 16, shuffle_true=False)

# model setting
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
model.eval()


def cosine_similarity(word1, word2):
    return np.dot(word1, word2) / (np.linalg.norm(word1)*np.linalg.norm(word2)+1e-10)


result1 = torch.Tensor([])
result2 = torch.Tensor([])
align_list = []
uniform_list = []

for idx, (data1, attn_mask1) in enumerate(loader1):
    # if idx == 0:
    #    print(data[0])
    with torch.no_grad():
        _, result_1 = model(
            data1.to(device), attention_mask=attn_mask1.to(device))
        #_, result_p = student_model(data.to(device), attention_mask=attn_mask.to(device))
        #align_list.append(align_loss(result, result_p))
        # uniform_list.append(uniform_loss(result))
        result1 = torch.cat((result1, result_1.detach().cpu()), dim=0)

for idx, (data2, attn_mask2) in enumerate(loader2):
    # if idx == 0:
    #    print(data[0])
    with torch.no_grad():
        _, result_2 = model(
            data2.to(device), attention_mask=attn_mask2.to(device))
        result2 = torch.cat((result2, result_2.detach().cpu()), dim=0)


cossim = []
for i in range(len(result1)):
    cossim.append(cosine_similarity(result1[i], result2[i]))
print("Spearman's Rank Correlation : ", spearmanr(
    labels, cossim, nan_policy='omit'))

# %%
