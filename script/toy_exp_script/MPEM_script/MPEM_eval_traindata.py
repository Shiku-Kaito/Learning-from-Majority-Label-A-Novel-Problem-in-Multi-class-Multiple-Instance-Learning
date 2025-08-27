
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import logging
from toy_exp_script.count_script.count_network import Count

from utils import *
from utils_analysis import org_prop_vs_acc_subtraction
from toy_exp_script.MPEM_script.PosBoost_func import gen_mask, gen_prototype,dist_intesrval_acc, prop_substraction_histgram_boxplot_pretrained_best_model, org_prop_boost_prop_scatter

def eval_net(args, model, test_loader):
    fix_seed(args.seed)
    result_dict = {}

################## test ###################
    s_time = time()
    # model.eval()
    ins_gt, bag_gt, ins_pred, ins_conf, bag_pred, bag_m, train_bag_idxs, train_inst_feat = [], [], [], [], [], [], [], []
    pretraind_model = Count(args.classes, args.temper1, args.temper2)
    pretraind_model = pretraind_model.to(args.device)
    pretraind_model.load_state_dict(torch.load(("./result/result_without_pretrain/%s/%s/count_T1=0.1_T2=0.1/model/fold=%d_seed=%d-best_model.pkl") % (args.dataset, args.majority_size, args.fold, args.seed) ,map_location=args.device))
    pretraind_model.eval()
    with torch.no_grad():
        for iteration, data in enumerate(test_loader): #enumerate(tqdm(test_loader, leave=False)):
            bag_label_copy=data["bag_label"].cpu().detach()
            ins_label, bag_label = data["ins_label"].reshape(-1), torch.eye(args.classes)[data["bag_label"]]
            bags, bag_label = data["bags"].to(args.device), bag_label.to(args.device)  

            y = pretraind_model(bags)

            ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label_copy.cpu().detach().numpy())
            ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())
            ins_conf.extend(y["ins"].cpu().detach().numpy())
            train_bag_idxs.extend(data["bag_idx"].cpu().detach().numpy())
            train_inst_feat.extend(y["ins_feat"].cpu().detach().numpy())

    ins_gt, bag_gt, ins_pred, bag_pred, ins_conf = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred), np.array(ins_conf)
    result_dict["ins_acc"] = (ins_gt == ins_pred).mean()
    result_dict["bag_acc"] = (bag_gt == bag_pred).mean()
    
    train_bag_idxs, train_inst_feat = np.array(train_bag_idxs), np.array(train_inst_feat)
    prototype, class_ins_feat_dict, empty_prototype_class = gen_prototype(args, ins_gt, ins_pred, bag_gt, bag_pred, train_inst_feat)
    mask_idx, removed_Ins_acc, removed_Ins_purity, removed_inst_num, boost_bag_acc  = gen_mask(args, ins_gt, ins_pred, train_inst_feat, bag_gt, bag_pred, train_bag_idxs, prototype, class_ins_feat_dict, empty_prototype_class)
    result_dict["removed_Ins_acc"]=removed_Ins_acc
    result_dict["removed_Ins_purity"]=removed_Ins_purity
    result_dict["removed_inst_num"]=removed_inst_num
    result_dict["boost_bag_acc"]=boost_bag_acc

    # prop_substraction_histgram_boxplot_pretrained_best_model(args, ins_gt, ins_pred, bag_gt, bag_pred,  "%s/substraction_boxplot/fold=%d_seed=%d_pretrain_bestepoch_model.png" % (args.output_path, args.fold, args.seed))
    # org_prop_boost_prop_scatter(args, ins_gt, ins_pred, bag_gt, bag_pred, mask_idx, train_bag_idxs)
    
    # print(removed_Ins_acc)
    # print(removed_inst_num)
    # print(boost_bag_acc)


    ############ train ###################
    s_time = time()
    boost_ins_gt, boost_bag_gt, boost_ins_pred, boost_ins_feat_pred, boost_boost_bag_labels, boost_bag_pred, boost_train_bag_idxs, train_mask_idx = [], [], [], [], [], [], [], []
    removed_inst_nums = 0
    model.eval()
    for iteration, data in enumerate(test_loader): #enumerate(tqdm(train_loader, leave=False)):
        bag_label_copy=data["bag_label"].cpu().detach()
        ins_label, bag_label = data["ins_label"].reshape(-1), torch.eye(args.classes)[data["bag_label"]]
        bags, bag_label = data["bags"].to(args.device), bag_label.to(args.device)   

        y = model(bags, mask_idx, data["bag_idx"])

        boost_ins_gt.extend(ins_label.cpu().detach().numpy()), boost_bag_gt.extend(bag_label_copy.cpu().detach().numpy())
        boost_ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), boost_bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())
        boost_train_bag_idxs.extend(data["bag_idx"].cpu().detach().numpy())
        train_mask_idx.extend(mask_idx)

    boost_ins_gt, boost_bag_gt, boost_ins_pred, boost_bag_pred, boost_train_bag_idxs = np.array(boost_ins_gt), np.array(boost_bag_gt), np.array(boost_ins_pred), np.array(boost_bag_pred), np.array(boost_train_bag_idxs)
    # org_prop_vs_acc_subtraction(args, ins_gt, ins_pred, bag_gt, train_bag_idxs,  boost_ins_gt, boost_ins_pred, boost_bag_gt, boost_train_bag_idxs, mask_idx)
    return result_dict