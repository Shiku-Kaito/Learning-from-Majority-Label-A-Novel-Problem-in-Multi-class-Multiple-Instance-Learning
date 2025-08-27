import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import logging
import copy
from utils import *
from toy_exp_script.MPEM_script.PosBoost_func import gen_mask, gen_prototype
from toy_exp_script.count_script.count_network import Count

def train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function):
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("%s/log_dict/fold=%d_seed=%d_training_setting.log" %  (args.output_path, args.fold, args.seed))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.info(args)
    print("mode:",args.mode, "temper1:", args.temper1, "temper2:", args.temper2)
    fix_seed(args.seed)
    log_dict = {"train_bag_acc":[], "train_ins_acc":[], "train_ins_feat_acc":[], "train_boos_bag_acc":[], "train_removed_ins_acc":[], "train_mIoU":[], "train_loss":[], "train_loss1":[],  "train_loss2":[], "majo_cls_recall":[], "majo_cls_precision":[], "pseude_ins_label_num":[],
                "val_bag_acc":[], "val_ins_acc":[], "val_mIoU":[], "val_loss":[], "val_loss1":[], "val_loss2":[], 
                "test_bag_acc":[], "test_ins_acc":[], "test_mIoU":[], "test_loss":[], 
                "flip_0_25":[], "flip_25_50":[], "flip_50_75":[], "flip_75_100":[]}
    
    best_val_loss = float('inf')
    cnt = 0
    prototype = []

    fix_seed(args.seed)
    result_dict = {}

################## pretrain model test ###################
    s_time = time()
    model.eval()
    ins_gt, bag_gt, ins_pred, ins_conf, bag_pred, bag_m, train_bag_idxs, train_inst_feat = [], [], [], [], [], [], [], []
    pretraind_model = Count(args.classes, args.temper1, args.temper2)
    pretraind_model = pretraind_model.to(args.device)
    pretraind_model.load_state_dict(torch.load(("./result/result_without_pretrain/%s/%s/count_T1=0.1_T2=0.1/model/fold=%d_seed=%d-best_model.pkl") % (args.dataset, args.majority_size, args.fold, args.seed) ,map_location=args.device))
    pretraind_model.eval()
    fix_seed(args.seed)
    with torch.no_grad():
        for iteration, data in enumerate(train_loader): #enumerate(tqdm(test_loader, leave=False)):
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
    logging.info('===================================================================================')
    logging.info('pretrain model train inst Accuracy: %.4f' % (ins_gt == ins_pred).mean())
    logging.info('===================================================================================')
    train_bag_idxs, train_inst_feat = np.array(train_bag_idxs), np.array(train_inst_feat)
    prototype, class_ins_feat_dict, empty_prototype_class = gen_prototype(args, ins_gt, ins_pred, bag_gt, bag_pred, train_inst_feat)
    mask_idx, removed_Ins_acc, removed_Ins_purity, removed_inst_num, boost_bag_acc  = gen_mask(args, ins_gt, ins_pred, train_inst_feat, bag_gt, bag_pred, train_bag_idxs, prototype, class_ins_feat_dict, empty_prototype_class)

    
    # fix_seed(args.seed)
    for epoch in range(args.num_epochs):
        train_int_conf, val_int_conf, test_int_conf, train_pred_prop, train_mask_idx, train_inst_feat = [], [], [], [], [], []
        ############ train ###################
        s_time = time()
        ins_gt, bag_gt, ins_pred, ins_feat_pred, boost_bag_labels, bag_pred, bag_m, losses, losses1, losses2, train_bag_idxs = [], [], [], [], [], [], [], [], [], [], []
        removed_inst_nums = 0
        model.train()
        for iteration, data in enumerate(train_loader): #enumerate(tqdm(train_loader, leave=False)):
            bag_label_copy=data["bag_label"].cpu().detach()
            ins_label, bag_label = data["ins_label"].reshape(-1), torch.eye(args.classes)[data["bag_label"]]
            bags, bag_label = data["bags"].to(args.device), bag_label.to(args.device)   

            y = model(bags, mask_idx, data["bag_idx"])
            
            loss = loss_function(y["bag"], bag_label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label_copy.cpu().detach().numpy())
            ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())
            losses.append(loss.item())
            train_int_conf.extend(y["ins_conf"].cpu().detach().numpy())
            train_pred_prop.extend(y["prop"].cpu().detach().numpy())
            train_bag_idxs.extend(data["bag_idx"].cpu().detach().numpy())
            # train_mask_idx.extend(mask_idx)
            train_inst_feat.extend(y["ins_feat"].cpu().detach().numpy())

        ins_gt, bag_gt, ins_pred, bag_pred, train_inst_feat, ins_feat_pred, boost_bag_labels = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred), np.array(train_inst_feat), np.array(ins_feat_pred), np.array(boost_bag_labels)
        log_dict["train_ins_acc"].append((ins_gt == ins_pred).mean()), log_dict["train_bag_acc"].append((bag_gt == bag_pred).mean())
        log_dict["train_loss"].append(np.array(losses).mean())

        train_cm = confusion_matrix(y_true=ins_gt, y_pred=ins_pred, normalize='true')
        log_dict["train_mIoU"].append(cal_mIoU(train_cm))

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] train loss: %.4f, ins acc: %.4f, bag acc:  %.4f, mIoU: %.4f,' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["train_loss"][-1], log_dict["train_ins_acc"][-1], log_dict["train_bag_acc"][-1], log_dict["train_mIoU"][-1]))

        
        ################# validation ####################
        s_time = time()
        model.eval()
        ins_gt, bag_gt, ins_pred, bag_pred, bag_m, losses, losses1, losses2, val_bag_idxs = [], [], [], [], [], [], [], [], []
        with torch.no_grad():
            for iteration, data in enumerate(val_loader): #enumerate(tqdm(val_loader, leave=False)):
                bag_label_copy=data["bag_label"].cpu().detach()
                ins_label, bag_label = data["ins_label"].reshape(-1), torch.eye(args.classes)[data["bag_label"]]
                bags, bag_label = data["bags"].to(args.device), bag_label.to(args.device)  

                y = model(bags, data["mask_idx"], None)
                loss = loss_function(y["bag"], bag_label)

                ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label_copy.cpu().detach().numpy())
                ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())
                losses.append(loss.item())
                val_int_conf.extend(y["ins_conf"].cpu().detach().numpy())
                val_bag_idxs.extend(data["bag_idx"].cpu().detach().numpy()) 

        ins_gt, bag_gt, ins_pred, bag_pred = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred)
        log_dict["val_ins_acc"].append((ins_gt == ins_pred).mean()), log_dict["val_bag_acc"].append((bag_gt == bag_pred).mean())
        log_dict["val_loss"].append(np.array(losses).mean())
        
        val_cm = confusion_matrix(y_true=ins_gt, y_pred=ins_pred, normalize='true')
        log_dict["val_mIoU"].append(cal_mIoU(val_cm))

        logging.info('[Epoch: %d/%d (%ds)] val loss: %.4f, ins acc: %.4f, bag acc:  %.4f, mIoU: %.4f' %
                        (epoch+1, args.num_epochs, e_time-s_time, log_dict["val_loss"][-1], log_dict["val_ins_acc"][-1], log_dict["val_bag_acc"][-1], log_dict["val_mIoU"][-1]))

        if args.is_test == True:
        ################## test ###################
            s_time = time()
            model.eval()
            ins_gt, bag_gt, ins_pred, bag_pred, bag_m = [], [], [], [], []
            with torch.no_grad():
                for iteration, data in enumerate(test_loader): #enumerate(tqdm(test_loader, leave=False)):
                    bag_label_copy=data["bag_label"].cpu().detach()
                    ins_label, bag_label = data["ins_label"].reshape(-1), torch.eye(args.classes)[data["bag_label"]]
                    bags, bag_label = data["bags"].to(args.device), bag_label.to(args.device)  

                    y = model(bags, data["mask_idx"], None)

                    ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label_copy.cpu().detach().numpy())
                    ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())
                    test_int_conf.extend(y["ins_conf"].cpu().detach().numpy())

            ins_gt, bag_gt, ins_pred, bag_pred = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred)
            log_dict["test_ins_acc"].append((ins_gt == ins_pred).mean()), log_dict["test_bag_acc"].append((bag_gt == bag_pred).mean()) 

            test_cm = confusion_matrix(y_true=ins_gt, y_pred=ins_pred, normalize='true')
            log_dict["test_mIoU"].append(cal_mIoU(test_cm))

            e_time = time()
            logging.info('[Epoch: %d/%d (%ds)] , ins acc: %.4f, bag acc: %.4f,  mIoU: %.4f' %
                            (epoch+1, args.num_epochs, e_time-s_time, log_dict["test_ins_acc"][-1], log_dict["test_bag_acc"][-1],  log_dict["test_mIoU"][-1]))
        logging.info('===============================')


        if best_val_loss > log_dict["val_loss"][-1]:
            best_val_loss = log_dict["val_loss"][-1]
            cnt = 0
            best_epoch = epoch
            torch.save(model.state_dict(), ("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed))
            save_confusion_matrix(cm=train_cm, path=("%s/cm/fold=%d_seed=%d-cm_train.png") % (args.output_path, args.fold, args.seed),
                        title='train: epoch: %d, acc: %.4f, mIoU: %.4f' % (epoch+1, log_dict["train_ins_acc"][epoch], log_dict["train_mIoU"][epoch]))
            save_confusion_matrix(cm=val_cm, path=("%s/cm/fold=%d_seed=%d-cm_val.png") % (args.output_path, args.fold, args.seed),
                        title='validation: epoch: %d, acc: %.4f, mIoU: %.4f' % (epoch+1, log_dict["val_ins_acc"][epoch], log_dict["val_mIoU"][epoch]))
            if args.is_test == True:
                save_confusion_matrix(cm=test_cm, path=("%s/cm/fold=%d_seed=%d-cm_test.png") % (args.output_path, args.fold, args.seed),
                            title='test: epoch: %d, acc: %.4f, mIoU: %.4f' % (epoch+1, log_dict["test_ins_acc"][epoch], log_dict["test_mIoU"][epoch]))
        else:
            cnt += 1
        
        logging.info('best epoch: %d, val bag acc: %.4f, val inst acc: %.4f, mIoU: %.4f' %
                        (best_epoch+1, log_dict["val_bag_acc"][best_epoch], log_dict["val_ins_acc"][best_epoch], log_dict["val_mIoU"][best_epoch]))
        if args.is_test == True:
            logging.info('best epoch: %d, test bag acc: %.4f, test inst acc: %.4f, mIoU: %.4f' %
                            (best_epoch+1, log_dict["test_bag_acc"][best_epoch], log_dict["test_ins_acc"][best_epoch], log_dict["test_mIoU"][best_epoch]))
            

        make_loss_graph(args,log_dict['train_loss'], log_dict['val_loss'], (args.output_path+ "/loss_graph/"+"/fold="+ str(args.fold) +"_seed="+str(args.seed)+"_loss_graph.png"))
        
        make_bag_acc_graph(args, log_dict['train_bag_acc'], log_dict['val_bag_acc'], (args.output_path+ "/acc_graph/"+"/fold="+ str(args.fold) +"_seed="+str(args.seed)+"_bag_acc_graph.png"))
        make_ins_acc_graph(args, log_dict['train_ins_acc'], log_dict['val_ins_acc'], (args.output_path+ "/acc_graph/"+"/fold="+ str(args.fold) +"_seed="+str(args.seed)+"_ins_acc_graph.png"))
        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
    return 
        
