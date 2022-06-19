import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from model import studentBertModel, BertConfig
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import RkdDistance, RKdAngle, ConLoss, align_loss, uniformity
from dataload import get_data_main, load_data


def RKD_Contrastive_Learning(args):

    # argument setting
    batch_size = args.bth

    if args.RKD_type == "distance":
        collect_num = 2
    elif args.RKD_type == "angle":
        collect_num = 3
    else:
        raise ValueError

    # Data preparation
    tokenized_data = get_data_main()
    #tokenized_data = tokenized_data.to(args.device)

    # Data Loader setting
    # get postive and negative samples
    loader = load_data(tokenized_data, args.device,
                       batch_size=batch_size*collect_num)

    # teacher model setting
    teacher_model = BertModel.from_pretrained('bert-base-uncased')
    teacher_model.to(args.device)
    teacher_model.eval()     # for only inference

    # student model setting
    parent_config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                               num_hidden_layers=6, num_attention_heads=1, intermediate_size=1536)
    student_model = studentBertModel(parent_config)
    student_model.to(args.device)
    student_model.train()

    # opimizer setting
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)

    # loss function setting
    if args.RKD_type == "distance":
        KD_loss = RkdDistance()
    elif args.RKD_type == "angle":
        KD_loss = RKdAngle()
    else:
        raise ValueError

    ctloss = ConLoss(device=args.device)

    for epoch in range(args.ep):
        epoch_sum_loss = 0
        epoch_ctl_loss = 0
        epoch_kd_loss = 0
        epoch_align_loss = 0
        epoch_uniform_loss = 0
        for idx, (data, attn_mask) in enumerate(loader):
            optimizer.zero_grad()
            # 적절히 slicing해서 distance case, angle-wise case 이쁘게 짜보기.. view를 사용하면 mini-batch들이 순서대로 묶임
            if idx == len(loader) - 1:
                break
            data_pairs = data.view(batch_size, collect_num, 512)
            attn_mask_pairs = attn_mask.view(batch_size, collect_num, 512)
            attn_mask_x_i = attn_mask_pairs[:, 0, :].to(args.device)
            attn_mask_x_j = attn_mask_pairs[:, 1, :].to(args.device)

            x_i = data_pairs[:, 0, :].to(args.device)
            x_j = data_pairs[:, 1, :].to(args.device)
            #
            # teach_model input : [batch_size, max_seq]
            _, t_i = teacher_model(x_i, attention_mask=attn_mask_x_i)
            _, t_j = teacher_model(x_j, attention_mask=attn_mask_x_j)
            #
            #teacher_pairs = sentence_emb.view(batch_size, collect_num, 768)

            # normalization
            t_i = t_i / torch.norm(t_i, dim=1).unsqueeze(1)
            t_j = t_j / torch.norm(t_j, dim=1).unsqueeze(1)
            #t_k = t_k / torch.norm(t_k, dim=1).unsqueeze(1)

            # s_i and s_i_p are sightly different due to the dropout
            _, s_i = student_model(x_i, attention_mask=attn_mask_x_i)
            _, s_i_p = student_model(x_i, attention_mask=attn_mask_x_i)
            _, s_i_n = student_model(x_j, attention_mask=attn_mask_x_j)

            # normalization
            s_i = s_i / torch.norm(s_i, dim=1).unsqueeze(1)
            s_i_p = s_i_p / torch.norm(s_i_p, dim=1).unsqueeze(1)
            s_i_n = s_i_n / torch.norm(s_i_n, dim=1).unsqueeze(1)

            if args.RKD_type == "angle":
                attn_mask_x_k = attn_mask_pairs[:, 2, :].to(args.device)
                x_k = data_pairs[:, 2, :].to(args.device)
                _, t_k = teacher_model(x_k, attention_mask=attn_mask_x_k)
                _, s_k = student_model(x_k, attention_mask=attn_mask_x_k)
                s_k = s_k / torch.norm(s_k, dim=1).unsqueeze(1)
                teacher_emb = torch.cat((t_i, t_j, t_k), dim=0)
                student_emb = torch.cat((s_i, s_i_n, s_k), dim=0)
            else:
                teacher_emb = torch.cat((t_i, t_j), dim=0)
                student_emb = torch.cat((s_i, s_i_n), dim=0)
            # KD_loss input : (student : [n_view, emdim], teacher : [n_view, emdim])
            # ctloss input : (positive pairs : [batch_size, 2 == (s_i, s_i_p), emb_dim], negative paris : [batch_size, 2 == (i, negative), emb_dim])
            #loss = KD_loss() + args.L * ctloss()
            pos_pair = torch.cat((s_i.unsqueeze(1), s_i_p.unsqueeze(1)), dim=1)
            neg_pair = torch.cat((s_i.unsqueeze(1), s_i_n.unsqueeze(1)), dim=1)

            ctl = ctloss(pos_pair, neg_pair)
            kdl = KD_loss(student_emb, teacher_emb)
            loss = kdl + args.L * ctl
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(
                    "{} epoch\t [{}/{}] iter\t loss : {}".format(epoch, idx, len(loader), loss.item()))
            epoch_sum_loss += loss.item()

            epoch_ctl_loss += ctl.item()
            epoch_kd_loss += kdl.item()
            epoch_align_loss += align_loss(s_i, s_i_p)
            epoch_uniform_loss += uniformity(s_i, s_i_n)
        epoch_mean_loss = epoch_sum_loss / len(loader)
        epoch_mean_ctl_loss = epoch_ctl_loss / len(loader)
        epoch_mean_kd_loss = epoch_kd_loss / len(loader)
        epoch_mean_align_loss = epoch_align_loss / len(loader)
        epoch_mean_uniform_loss = epoch_uniform_loss / len(loader)

        f = open(args.REPORT_PATH + '/report.txt', "a")
        f.write("epoch : {}\tloss : {}\tctl : {}\tkd : {}\talign : {}\tuniform : {}\n".format(
            epoch, epoch_mean_loss, epoch_mean_ctl_loss, epoch_mean_kd_loss, epoch_mean_align_loss, epoch_mean_uniform_loss))
        f.close()
        torch.save(student_model.state_dict(), args.REPORT_PATH +
                   '/model_{}.pth'.format(epoch))


def Contrastive_Learning(args):

    # argument setting
    batch_size = args.bth

    if args.RKD_type == "distance":
        collect_num = 2
    elif args.RKD_type == "angle":
        collect_num = 3
    else:
        raise ValueError

    # Data preparation
    tokenized_data = get_data_main()
    #tokenized_data = tokenized_data.to(args.device)

    # Data Loader setting
    # get postive and negative samples
    loader = load_data(tokenized_data, args.device,
                       batch_size=batch_size*collect_num)

    # student model setting
    parent_config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                               num_hidden_layers=6, num_attention_heads=1, intermediate_size=1536)
    student_model = studentBertModel(parent_config)
    student_model.to(args.device)
    student_model.train()

    # opimizer setting
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)

    # loss function setting
    ctloss = ConLoss(device=args.device)

    for epoch in range(args.ep):
        epoch_sum_loss = 0
        epoch_ctl_loss = 0
        epoch_align_loss = 0
        epoch_uniform_loss = 0
        for idx, (data, attn_mask) in enumerate(loader):
            optimizer.zero_grad()
            # 적절히 slicing해서 distance case, angle-wise case 이쁘게 짜보기.. view를 사용하면 mini-batch들이 순서대로 묶임
            if idx == len(loader) - 1:
                break
            data_pairs = data.view(batch_size, collect_num, 512)
            attn_mask_pairs = attn_mask.view(batch_size, collect_num, 512)
            attn_mask_x_i = attn_mask_pairs[:, 0, :].to(args.device)
            attn_mask_x_j = attn_mask_pairs[:, 1, :].to(args.device)

            x_i = data_pairs[:, 0, :].to(args.device)
            x_j = data_pairs[:, 1, :].to(args.device)

            # s_i and s_i_p are sightly different due to the dropout
            _, s_i = student_model(x_i, attention_mask=attn_mask_x_i)
            _, s_i_p = student_model(x_i, attention_mask=attn_mask_x_i)
            _, s_i_n = student_model(x_j, attention_mask=attn_mask_x_j)

            # normalization
            s_i = s_i / torch.norm(s_i, dim=1).unsqueeze(1)
            s_i_p = s_i_p / torch.norm(s_i_p, dim=1).unsqueeze(1)
            s_i_n = s_i_n / torch.norm(s_i_n, dim=1).unsqueeze(1)

            # KD_loss input : (student : [n_view, emdim], teacher : [n_view, emdim])
            # ctloss input : (positive pairs : [batch_size, 2 == (s_i, s_i_p), emb_dim], negative paris : [batch_size, 2 == (i, negative), emb_dim])
            pos_pair = torch.cat((s_i.unsqueeze(1), s_i_p.unsqueeze(1)), dim=1)
            neg_pair = torch.cat((s_i.unsqueeze(1), s_i_n.unsqueeze(1)), dim=1)

            ctl = ctloss(pos_pair, neg_pair)
            loss = ctl
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(
                    "{} epoch\t [{}/{}] iter\t loss : {}".format(epoch, idx, len(loader), loss.item()))
            epoch_sum_loss += loss.item()

            epoch_ctl_loss += ctl.item()
            epoch_align_loss += align_loss(s_i, s_i_p)
            epoch_uniform_loss += uniformity(s_i, s_i_n)
        epoch_mean_loss = epoch_sum_loss / len(loader)
        epoch_mean_ctl_loss = epoch_ctl_loss / len(loader)
        epoch_mean_align_loss = epoch_align_loss / len(loader)
        epoch_mean_uniform_loss = epoch_uniform_loss / len(loader)

        f = open(args.REPORT_PATH + '/report.txt', "a")
        f.write("epoch : {}\tloss : {}\tctl : {}\talign : {}\tuniform : {}\n".format(
            epoch, epoch_mean_loss, epoch_mean_ctl_loss, epoch_mean_align_loss, epoch_mean_uniform_loss))
        f.close()
        torch.save(student_model.state_dict(), args.REPORT_PATH +
                   '/model_{}.pth'.format(epoch))
