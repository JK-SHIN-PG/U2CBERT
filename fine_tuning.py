import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from model import studentBertModel, BertConfig, sBertClassifier
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import RkdDistance, RKdAngle, ConLoss, align_loss, uniform_loss
from dataload import get_data, load_data, construct_data_for_finetuning


def train(model, loader, optimizer, criterion, args):
    ep_loss = 0
    model.train()
    for idx, (data, attn_mask, token_ids, label) in enumerate(loader):
        optimizer.zero_grad()
        out = model(data.to(args.device), attention_mask=attn_mask.to(args.device),
                    token_type_ids=token_ids.to(args.device))
        if args.f_task == "sick":
            out = F.softmax(out, dim=1)
            loss = criterion(out.squeeze(), label.to(args.device))
        else:
            loss = criterion(out.squeeze(), label.type(
                torch.float).to(args.device))
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
    return ep_loss / len(loader)


def valid(model, loader, criterion, args):
    ep_loss = 0
    ep_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, (data, attn_mask, token_ids, label) in enumerate(loader):
            out = model(data.to(args.device), attention_mask=attn_mask.to(args.device),
                        token_type_ids=token_ids.to(args.device))
            if args.f_task == "sick":
                out = F.softmax(out, dim=1)
                pred = torch.argmax(out, dim=1)
                loss = criterion(out.squeeze(), label.to(args.device))
            else:
                pred = torch.round(out).type(torch.long)
                loss = criterion(out.squeeze(), label.type(
                    torch.float).to(args.device))

            acc = (torch.sum(pred.squeeze().detach().clone().cpu()
                             == label) / label.shape[0]) * 100
            ep_loss += loss.item()
            ep_acc += acc.item()
    return ep_loss / len(loader), ep_acc / len(loader)


def Fine_Tuning(args):
    # data load
    t_data1, t_data2, t_labels = get_data(data_type="train", task=args.f_task)
    v_data1, v_data2, v_labels = get_data(data_type="valid", task=args.f_task)

    # fine tunning datasets construction
    t_data, t_token_ids = construct_data_for_finetuning(t_data1, t_data2)
    v_data, v_token_ids = construct_data_for_finetuning(v_data1, v_data2)

    t_loader = load_data(t_data, args.device, args.bth, shuffle_true=False,
                         ft=True, token_ids=t_token_ids, labels=t_labels)
    v_loader = load_data(v_data, args.device, args.bth, shuffle_true=False,
                         ft=True, token_ids=v_token_ids, labels=v_labels)

    # model load
    parent_config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                               num_hidden_layers=6, num_attention_heads=1, intermediate_size=1536)
    model = studentBertModel(parent_config)
    MODEL_PATH = args.save
    model.load_state_dict(torch.load(
        "./save_file/" + MODEL_PATH + '/model_43.pth', map_location=args.device))

    if args.f_task == "sick":
        classifier = sBertClassifier(model, hidden=64, out_dim=3)
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.f_task == "mrpc":
        classifier = sBertClassifier(model, hidden=64, out_dim=1)
        criterion = nn.BCEWithLogitsLoss().to(args.device)
    else:
        pass

    classifier.to(args.device)

    # optimizer setting
    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, eps=1e-8)

    for ep in range(args.ep):
        t_loss = train(classifier, t_loader, optimizer, criterion, args)
        print("ep : {}\ttrain loss : {}".format(ep+1, t_loss))
        if (ep+1) % 10 == 0:
            v_loss, v_acc = valid(classifier, v_loader, criterion, args)
            print("ep : {}\tvalid loss : {}\tvaild acc : {}".format(
                ep+1, v_loss, v_acc))

            torch.save(classifier.state_dict(), args.REPORT_PATH +
                       '/tf_model_{}_{}.pth'.format(ep+1, args.f_task))
        f = open(args.REPORT_PATH + '/ft_report_{}.txt'.format(args.f_task), "a")
        f.write("epoch : {}\tloss : {}\n".format(ep+1, t_loss))
        f.close()
