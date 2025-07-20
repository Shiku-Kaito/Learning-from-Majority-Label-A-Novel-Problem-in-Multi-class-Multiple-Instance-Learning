import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from get_module import get_module


def main(args):
    fix_seed(args.seed)
    train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader = get_module(args)
    args.output_path += '%s/%s/%s' % (args.dataset, args.majority_size, args.mode) 
    make_folder(args)

    model.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))
    args.eval_data = "test"
    result_dict_test = eval_net(args, model, test_loader, loss_function)
    args.eval_data = "validaton" 
    result_dict_val = eval_net(args, model, val_loader, loss_function)
    
    return result_dict_test, result_dict_val

if __name__ == '__main__':
    results_5fold = {"test_ins_acc":[], "test_bag_acc":[], "val_loss": [], "val_bag_acc": []}
    for non_pos_mask_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        results_dict_test = {"bag_acc":[], "ins_acc":[], "consistency_rate":[]}
        results_dict_validaton = {"losses":[], "bag_acc":[]}
        for fold in range(5):
        # for seed in range(1):
            parser = argparse.ArgumentParser()
            # Data selectiion
            parser.add_argument('--fold', default=fold,
                                type=int, help='fold number')
            parser.add_argument('--dataset', default='cifar10',
                                type=str, help='cifar10 or mnist or svhn or blood or oct or organa or organc or organs or path')
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
            parser.add_argument('--is_evaluation', default=1,
                                type=int, help='1 or 0')                             
            # Module Selection
            parser.add_argument('--module',default='MPEM', 
                                type=str, help="")
            parser.add_argument('--mode',default='',
                                type=str, help="")                        
            # Save Path
            parser.add_argument('--output_path',
                                default='./result/result_without_pretrain/', type=str, help="output file name")
            ### Module detail ####
            # Count Parameter
            parser.add_argument('--temper1', default=0.1,
                                type=float, help='softmax temper of before counting')
            parser.add_argument('--temper2', default=0.1,
                                type=float, help='softmax temper of after counting')  
            # MPEM Parameter
            parser.add_argument('--non_pos_mask_rate', default=non_pos_mask_rate, type=float, help='')   
            parser.add_argument('--feat_dist', default="euclidean", type=str, help='euclidean or mahalanobis')  
            parser.add_argument('--eval_data', default="validaton", type=str, help='test or validaton')  
            args = parser.parse_args()  
            

            result_dict_test, result_dict_val = main(args)
            results_dict_test["bag_acc"].append(result_dict_test["bag_acc"]), results_dict_test["ins_acc"].append(result_dict_test["ins_acc"]), results_dict_test["consistency_rate"].append(result_dict_test["consistency_rate"])
            results_dict_validaton["losses"].append(result_dict_val["val_loss"]), results_dict_validaton["bag_acc"].append(result_dict_val["bag_acc"])

        results_5fold["test_ins_acc"].append(results_dict_test["ins_acc"])
        results_5fold["test_bag_acc"].append(results_dict_test["bag_acc"])
        results_5fold["val_loss"].append(results_dict_validaton["losses"])
        results_5fold["val_bag_acc"].append(results_dict_validaton["bag_acc"])

        print("5 fold cross validation k=%.2f, Inst acc: %.5f"% (args.non_pos_mask_rate, np.mean(np.array(results_dict_test["ins_acc"]))))
        
    best_k_indices = np.argmin(results_5fold["val_loss"], axis=0)
    ins_acc_list, bag_acc_list, val_loss_list = [], [], []
    for fold in range(5):
        ins_acc_list.append(results_5fold["test_ins_acc"][best_k_indices[fold]][fold])
        bag_acc_list.append(results_5fold["test_bag_acc"][best_k_indices[fold]][fold])
        val_loss_list.append(results_5fold["val_loss"][best_k_indices[fold]][fold])
    print("=====================================================================================")
    print("Based validation loss")        
    print("5 fold cross validation, Inst acc: %.3f$\pm$%.3f, Bag acc: %.4f±%.4f, validation loss: %.4f±%.4f" % (np.mean(np.array(ins_acc_list)), np.std(np.array(ins_acc_list)), np.mean(np.array(bag_acc_list)), np.std(np.array(bag_acc_list)), np.mean(np.array(val_loss_list)), np.std(np.array(val_loss_list))))
    print("loss selected k:", (best_k_indices+1)*10)
    print("=====================================================================================")
