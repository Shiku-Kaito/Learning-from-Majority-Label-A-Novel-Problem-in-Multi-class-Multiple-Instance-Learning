
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import logging

from utils import *


def eval_net(args, model, test_loader, loss_function):
    fix_seed(args.seed)
    result_dict = {}

################## test ###################
    s_time = time()
    model.eval()
    ins_gt, bag_gt, ins_pred, ins_conf, bag_pred, bag_m, losses = [], [], [], [], [], [], []
    
    with torch.no_grad():
        for iteration, data in enumerate(test_loader): #enumerate(tqdm(test_loader, leave=False)):
            bag_label_copy=data["bag_label"].cpu().detach()
            ins_label, bag_label = data["ins_label"].reshape(-1), torch.eye(args.classes)[data["bag_label"]]
            bags, bag_label = data["bags"].to(args.device), bag_label.to(args.device)  

            y = model(bags, data["mask_idx"], None)
            if args.eval_data == "validaton": 
                loss = loss_function(y["bag"], bag_label)

            ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label_copy.cpu().detach().numpy())
            ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())
            ins_conf.extend(y["ins"].cpu().detach().numpy())
            if args.eval_data == "validaton": 
                losses.append(loss.item())
            
    ins_gt, bag_gt, ins_pred, bag_pred, ins_conf = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred), np.array(ins_conf)
    result_dict["ins_acc"] = (ins_gt == ins_pred).mean()
    result_dict["bag_acc"] = (bag_gt == bag_pred).mean()
    if args.eval_data == "validaton": 
        result_dict["val_loss"] = (np.array(losses).mean())
    
    # calcurate consistenscy rate
    ins_pred = np.eye(args.classes)[ins_pred]
    ins_pred = ins_pred.reshape(-1, args.bag_size, args.classes)
    ins_pred = ins_pred[bag_gt == bag_pred]
    ins_conf = ins_conf.reshape(-1, args.bag_size, args.classes)
    ins_conf = ins_conf[bag_gt == bag_pred]
    
    ins_pred_count, ins_conf_sum = ins_pred.sum(1), ins_conf.sum(1)
    ins_pred_count_max, ins_conf_sum_max = ins_pred_count.argmax(1), ins_conf_sum.argmax(1)
    result_dict["consistency_rate"] = (ins_pred_count_max == ins_conf_sum_max).mean()
    
    print(result_dict["ins_acc"] )
    return result_dict
        
