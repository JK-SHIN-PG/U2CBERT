# %%
import torch
from pytorch_pretrained_bert import BertModel
from model import BertClassifier
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import RkdDistance, RKdAngle, ConLoss, align_loss, uniform_loss
from dataload import get_data, load_data, construct_data_for_finetuning
import os
from torchmetrics import F1Score
# device setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_num = "0"
device = torch.device(
    f"cuda:{gpu_num}") if torch.cuda.is_available() else torch.device("cpu")

f_task = "sick"
t_data1, t_data2, t_labels = get_data(data_type="test", task=f_task)
t_data, t_token_ids = construct_data_for_finetuning(t_data1, t_data2)
t_loader = load_data(t_data, device, 16, shuffle_true=False,
                     ft=True, token_ids=t_token_ids, labels=t_labels)


model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
MODEL_PATH = "bert-fine-tuning"
#model.load_state_dict(torch.load("./save_file/" + MODEL_PATH + '/model_43.pth', map_location=device))

if f_task == "sick":
    classifier = BertClassifier(model, hidden=64, out_dim=3)
elif f_task == "mrpc":
    classifier = BertClassifier(model, hidden=64, out_dim=1)
else:
    pass

ep = 300
classifier.load_state_dict(torch.load(
    "./save_file/" + MODEL_PATH + '/tf_model_{}_{}.pth'.format(ep, f_task), map_location=device))

classifier.to(device)
classifier.eval()

true_tensor = torch.Tensor([])
pred_tensor = torch.Tensor([])
with torch.no_grad():
    for idx, (data, attn_mask, token_ids, label) in enumerate(t_loader):
        out = classifier(data.to(device), attention_mask=attn_mask.to(device),
                         token_type_ids=token_ids.to(device))
        if f_task == "sick":
            out = F.softmax(out, dim=1)
            pred = torch.argmax(out, dim=1)
        else:
            pred = torch.round(out).type(torch.long)
        pred_tensor = torch.cat(
            (pred_tensor, pred.squeeze().detach().clone().cpu()))
        true_tensor = torch.cat((true_tensor, label))


if f_task == "sick":
    f1 = F1Score(num_classes=3)
    f1score = f1(pred_tensor.type(torch.long), true_tensor.type(torch.long))
elif f_task == "mrpc":
    f1 = F1Score()
    f1score = f1(pred_tensor, true_tensor.type(torch.long))

acc = (torch.sum(pred_tensor == true_tensor.type(
    torch.long)) / true_tensor.shape[0]) * 100
print("F1 score : ", f1score)
print("accuracy : ", acc)
# %%
