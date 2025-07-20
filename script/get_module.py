import argparse
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from dataloader  import *

from toy_exp_script.count_script.count_train import train_net as Count_train_net
from toy_exp_script.count_script.eval import eval_net as Count_eval_net
from toy_exp_script.count_script.count_network import Count
from losses import cross_entropy_loss

from toy_exp_script.MPEM_script.PosBoost_func import load_org_traindata
from toy_exp_script.MPEM_script.count_train import train_net as MPEM_train_net
from toy_exp_script.MPEM_script.eval import eval_net as MPEM_eval_net
from toy_exp_script.MPEM_script.count_network import MPEM_count_model



def get_module(args):
    if args.module ==  "Count":
        args.mode = "count_T1=%s_T2=%s" % (str(args.temper1), str(args.temper2))
        # Dataloader
        train_loader, val_loader, test_loader = load_data_bags(args)        
        # Model
        model = Count(args.classes, args.temper1, args.temper2)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = cross_entropy_loss 
        # Train net
        train_net = Count_train_net
        eval_net = Count_eval_net
        

    elif args.module ==  "MPEM":
        args.mode = "/MPEM_T1=%s_T2=%s_remove_thresh=%.2f_feat_dist=%s" % (str(args.temper1),str(args.temper2), args.non_pos_mask_rate, args.feat_dist)  

        train_loader, val_loader, test_loader = load_org_traindata(args)        
        model = MPEM_count_model(args.classes, args.temper1, args.temper2)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = cross_entropy_loss   
        # Train net
        train_net = MPEM_train_net
        eval_net = MPEM_eval_net

    else:
        print("Module ERROR!!!!!")

    return train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader
