import torch
import sys
import os
import argparse
from utils import ensure_path
from train import *
from fine_tuning_base_bert import *


class arg_init:
    def __init__(self, params):
        self.gpu_num = params.gpu
        self.ep = params.ep
        self.bth = params.bth
        self.lr = params.lr
        self.L = params.L
        self.save = params.save
        self.RKD_type = params.RKD_tp
        self.p_train = params.p_train
        self.CL = params.CL
        self.f_train = params.f_train
        self.f_task = params.f_task
        self.MODEL_STORAGE_PATH = "./save_file/"
        self.REPORT_PATH = self.MODEL_STORAGE_PATH + self.save
        self.reporting(params)
        self.device = torch.device(
            f"cuda:{self.gpu_num}") if torch.cuda.is_available() else torch.device("cpu")

    def reporting(self, params):
        # ensure_path(self.REPORT_PATH)
        f = open(self.REPORT_PATH + '/ft_report.txt', "w")
        f.write(str(vars(params)) + "\n")
        f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='3')
    parser.add_argument("--ep", type=int, default=300)
    parser.add_argument("--bth", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--L", type=float, default=0.001)
    parser.add_argument("--save", default='bert-fine-tuning')
    parser.add_argument("--RKD_tp", type=str, default="distance")
    parser.add_argument("--p_train", type=str, default="False")
    parser.add_argument("--CL", type=str, default="False")
    parser.add_argument("--f_train", type=str, default="False")
    parser.add_argument("--f_task", type=str, default="sick")
    parsed_args = parser.parse_args()

    args = arg_init(parsed_args)

    Fine_Tuning(args)
