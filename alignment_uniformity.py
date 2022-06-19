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
from utils import align_loss, uniformity
# device setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_num = "0"
device = torch.device(
    f"cuda:{gpu_num}") if torch.cuda.is_available() else torch.device("cpu")

# data setting
t_data1, t_data2, t_labels = get_data(data_type="train", task="mrpc")

tdata1 = t_data1[torch.Tensor(t_labels) == 1]
tdata2 = t_data2[torch.Tensor(t_labels) == 1]
tdata3 = t_data2[torch.Tensor(t_labels) == 0]
diff_data = torch.cat((tdata3, tdata3, tdata3[:86]))

loader1 = load_data(tdata1, device, 16, shuffle_true=False)
loader2 = load_data(tdata2, device, 16, shuffle_true=False)
loader3 = load_data(diff_data, device, 16, shuffle_true=False)

# model setting

parent_config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                           num_hidden_layers=6, num_attention_heads=1, intermediate_size=1536)
student_model = studentBertModel(parent_config)

MODEL_PATH = "CL_distance"
student_model.load_state_dict(torch.load(
    "./save_file/" + MODEL_PATH + '/model_99.pth', map_location=device))
student_model.to(device)
student_model.eval()

'''
student_model = BertModel.from_pretrained('bert-base-uncased')
student_model.to(device)
student_model.eval()
'''
result1 = torch.Tensor([])
result2 = torch.Tensor([])
result3 = torch.Tensor([])
align_list = []
uniform_list = []

for idx, (data1, attn_mask1) in enumerate(loader1):
    # if idx == 0:
    #    print(data[0])
    with torch.no_grad():
        _, result_1 = student_model(
            data1.to(device), attention_mask=attn_mask1.to(device))
        #_, result_p = student_model(data.to(device), attention_mask=attn_mask.to(device))
        #align_list.append(align_loss(result, result_p))
        # uniform_list.append(uniform_loss(result))
        result1 = torch.cat((result1, result_1.detach().cpu()), dim=0)

for idx, (data2, attn_mask2) in enumerate(loader2):
    # if idx == 0:
    #    print(data[0])
    with torch.no_grad():
        _, result_2 = student_model(
            data2.to(device), attention_mask=attn_mask2.to(device))
        result2 = torch.cat((result2, result_2.detach().cpu()), dim=0)

for idx, (data3, attn_mask3) in enumerate(loader3):
    # if idx == 0:
    #    print(data[0])
    with torch.no_grad():
        _, result_3 = student_model(
            data3.to(device), attention_mask=attn_mask3.to(device))
        result3 = torch.cat((result3, result_3.detach().cpu()), dim=0)

alignment = align_loss(result1, result2)
uniform = uniformity(result1, result3)
print("alignment : {}".format(alignment))
print("uniform : {}".format(uniform))

# %%
