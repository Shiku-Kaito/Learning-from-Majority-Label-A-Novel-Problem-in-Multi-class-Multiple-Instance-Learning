import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from get_module import get_module
from time import time


def main(args):
    fix_seed(args.seed)
    train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader = get_module(args)
    args.output_path += '%s/%s/%s' % (args.dataset, args.majority_size, args.mode) 
    make_folder(args)

    if args.is_evaluation == False:
        train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function)
        return
    elif args.is_evaluation == True:
        model.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))
        result_dict = eval_net(args, model, test_loader)
        return result_dict



if __name__ == '__main__':
    results_dict = {"bag_acc":[], "ins_acc":[], "consistency_rate":[], "losses":[], "time":[]}
    for fold in range(5):
        parser = argparse.ArgumentParser()
        # Data selectiion
        parser.add_argument('--fold', default=fold,
                            type=int, help='fold number')
        parser.add_argument('--dataset', default="cifar10",
                            type=str, help='cifar10 or svhn or oct or path')
        parser.add_argument('--classes', #書き換え
                            default=10, type=int, help="number of the sampled instnace")
        parser.add_argument('--majority_size', #書き換え
                            default="various", type=str, help="_small or  or _large")
        parser.add_argument('--bag_size', default=64, type=int, help="")
        # Training Setup
        parser.add_argument('--num_epochs', default=1500, type=int,
                            help='number of epochs for training.')
        parser.add_argument('--device', default='cuda:0',
                            type=str, help='device')
        parser.add_argument('--batch_size', default=64,
                            type=int, help='batch size for training.')
        parser.add_argument('--seed', default=0,
                            type=int, help='seed value')
        parser.add_argument('--num_workers', default=0, type=int,
                            help='number of workers for training.')
        parser.add_argument('--patience', default=100,
                            type=int, help='patience of early stopping')
        parser.add_argument('--lr', default=3e-4,
                            type=float, help='learning rate')
        parser.add_argument('--is_test', default=1,
                            type=int, help='1 or 0')      
        parser.add_argument('--is_evaluation', default=0,
                            type=int, help='1 or 0')                             
        # Module Selection
        parser.add_argument('--module',default='MPEM', 
                            type=str, help="Count or MPEM")
        parser.add_argument('--mode',default='',
                            type=str, help="")                        
        # Save Path
        parser.add_argument('--output_path',
                            default='./result/result_without_pretrain/', type=str, help="output file name")
        ### Module detail ####
        # Count Parameter
        parser.add_argument('--temper1', default=0.1,
                            type=int, help='softmax temper of before counting')
        parser.add_argument('--temper2', default=0.1,
                            type=float, help='softmax temper of after counting')  
        # pos boost
        parser.add_argument('--non_pos_mask_rate', default=0.1, type=float, help='')   
        parser.add_argument('--feat_dist', default="euclidean", type=str, help='euclidean or mahalanobis')  
        parser.add_argument('--eval_data', default="test", type=str, help='test or validaton')  
        args = parser.parse_args()  
        
        if args.is_evaluation == False:
            main(args)

        else:
            result_dict = main(args)
            results_dict["bag_acc"].append(result_dict["bag_acc"]), results_dict["ins_acc"].append(result_dict["ins_acc"]), results_dict["consistency_rate"].append(result_dict["consistency_rate"])

    if args.is_evaluation == True:
        print("=====================================================================================")    
        print("5 fold cross validation, Inst acc: %.6f±%.6f, Bag acc: %.4f±%.6f,, consistency rate: %.4f±%.6f," % (np.mean(np.array(results_dict["ins_acc"])), np.std(np.array(results_dict["ins_acc"])), np.mean(np.array(results_dict["bag_acc"])), np.std(np.array(results_dict["bag_acc"])), np.mean(np.array(results_dict["consistency_rate"])), np.std(np.array(results_dict["consistency_rate"]))))
        print("=====================================================================================")
        