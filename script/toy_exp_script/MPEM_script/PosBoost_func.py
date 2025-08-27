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
from scipy.spatial.distance import mahalanobis

class ORG_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, bags, ins_label, major_labels):
        np.random.seed(args.seed)
        self.bags = bags
        self.ins_label = ins_label
        self.major_labels = major_labels
        # self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.classes = args.classes
        self.len = len(self.bags)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        bag = self.bags[idx]
        (b, w, h, c) = bag.shape
        trans_bag = torch.zeros((b, c, w, h))
        for i in range(b):
            trans_bag[i] = self.transform(bag[i])
            
        ins_label = self.ins_label[idx]
        ins_label = torch.tensor(ins_label).long()
        major_label = self.major_labels[idx]
        major_label = torch.tensor(major_label).long() 
        return {"bags": trans_bag, "ins_label": ins_label, "bag_label": major_label, "bag_idx":idx, "len_list":len(ins_label), "mask_idx":[]}

def load_org_traindata(args):
    train_bags = np.load('./data/%s/%dclass_%s/%d/train_bags.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_ins_labels = np.load('./data/%s/%dclass_%s/%d/train_ins_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_major_labels = np.load('./data/%s/%dclass_%s/%d/train_major_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_dataset = ORG_Dataset(args=args, bags=train_bags, ins_label=train_ins_labels, major_labels=train_major_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, collate_fn=collate_fn_custom)

    test_data = np.load('./data/%s/%dclass_%s/%d/test_bags.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    test_ins_labels = np.load('./data/%s/%dclass_%s/%d/test_ins_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    test_major_labels = np.load('./data/%s/%dclass_%s/%d/test_major_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    test_dataset = ORG_Dataset(args=args, bags=test_data, ins_label=test_ins_labels, major_labels=test_major_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom)

    val_bags = np.load('./data/%s/%dclass_%s/%d/val_bags.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    val_ins_labels = np.load('./data/%s/%dclass_%s/%d/val_ins_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    val_major_labels = np.load('./data/%s/%dclass_%s/%d/val_major_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    val_dataset = ORG_Dataset(args=args, bags=val_bags, ins_label=val_ins_labels, major_labels=val_major_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn_custom)   
    return train_loader, val_loader, test_loader
    
def collate_fn_custom(batch):
    bags, ins_labels, bag_labels, bag_idxs, len_list, remove_ins_idxs = [], [], [], [], [], []
    for b in batch:
        bags.append(b["bags"]), ins_labels.append(b["ins_label"]), bag_labels.append(b["bag_label"]), bag_idxs.append(b["bag_idx"]), len_list.append(b["len_list"]), remove_ins_idxs.append(b["mask_idx"])

    bags, ins_labels, bag_labels, bag_idxs = torch.stack(bags, dim=0), torch.stack(ins_labels, dim=0), torch.stack(bag_labels, dim=0), torch.tensor(bag_idxs)
    return {"bags": bags, "ins_label": ins_labels, "bag_label": bag_labels, "bag_idx": bag_idxs, "len_list":len_list, "mask_idx":remove_ins_idxs}


def culc_mahalanobis_dist(class_ins_feat_dict, train_inst_feats):
    # クラスごとに重心と逆共分散行列を計算
    class_statistics = {}
    for class_name, data in class_ins_feat_dict.items():
        data = np.array(data)
        mean_vector = np.mean(data, axis=0)
        cov_matrix = np.cov(data, rowvar=False)
        # 正則化項を加える
        # cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        cov_inv = np.linalg.pinv(cov_matrix)
        class_statistics[class_name] = {
            "mean_vector": mean_vector,
            "cov_inv": cov_inv
        }

    # 各テスト点に対するクラスごとのマハラノビス距離を計算
    distances = []
    for train_inst_feat in train_inst_feats:
        class_distances = []
        for class_name, stats in class_statistics.items():
            mean_vector = stats["mean_vector"]
            cov_inv = stats["cov_inv"]
            distance = mahalanobis(train_inst_feat, mean_vector, cov_inv)
            class_distances.append(distance)
        distances.append(class_distances)   
    return np.array(distances)

def gen_mask(args, train_inst_gt, train_inst_pred, train_inst_feat, train_bag_gt, train_bag_pred, train_bag_idxs, prototype, class_ins_feat_dict, empty_prototype_class):
    bag_label_list, ins_gt_list = [], []
    mask_dict = {}
    
    # culcirate Euclidean distance
    train_inst_feat = train_inst_feat.reshape(-1, 512)
    train_inst_gt, train_inst_pred = train_inst_gt.reshape(-1, args.bag_size), train_inst_pred.reshape(-1, args.bag_size)
    if args.feat_dist == "euclidean":
        train_inst_distances = np.linalg.norm(train_inst_feat[:, np.newaxis] - prototype, axis=2)
    elif args.feat_dist == "mahalanobis":
        train_inst_distances = culc_mahalanobis_dist(class_ins_feat_dict, train_inst_feat)

    feat_nearest_class =  np.argmin(train_inst_distances, axis=1)
    feat_nearest_class = feat_nearest_class.reshape(-1, args.bag_size)

    # logging.info('===================================================================================')
    # logging.info('Prototype nearest class Accuracy: %.4f' % (np.array(feat_nearest_class.reshape(-1))==np.array(train_inst_gt.reshape(-1))).mean())
    
    train_inst_distances = train_inst_distances.reshape(-1, args.bag_size, args.classes)
    other_cls_inst_distances_list, prototype_cls_list = [], []
    pred_minor_inst_distances_list, pred_minor_inst_gt_list, pred_minor_prototype_cls_list = [], [], []
    ins_idx = np.arange(args.bag_size)
    for idx in range(len(train_bag_gt)):
        if train_bag_gt[idx] in empty_prototype_class:
            mask_dict[train_bag_idxs[idx]] = []
            continue
        other_cls_inst_distances = train_inst_distances[idx][:,train_bag_gt[idx]]
        other_cls_inst_distances_list.extend(other_cls_inst_distances), prototype_cls_list.extend([train_bag_gt[idx]]*len(train_inst_gt[idx]))
        idx_other_cls_inst = ins_idx[(train_inst_pred[idx]!=train_bag_gt[idx])]
        other_cls_inst_distances = other_cls_inst_distances[idx_other_cls_inst]
        pred_minor_inst_distances_list.extend(other_cls_inst_distances), pred_minor_prototype_cls_list.extend([train_bag_gt[idx]]*len(train_inst_gt[idx][idx_other_cls_inst])),  pred_minor_inst_gt_list.extend(train_inst_gt[idx][idx_other_cls_inst])
        sorted_indices_other_cls_inst_distances = np.argsort(other_cls_inst_distances)[::-1]
        majo_count = sum(train_inst_pred[idx]==train_bag_gt[idx])
        remove_inst_idx = idx_other_cls_inst[sorted_indices_other_cls_inst_distances[:int(((args.bag_size) - majo_count) * args.non_pos_mask_rate)]]
        
        preseud_ins_gt = train_inst_gt[idx][remove_inst_idx]
        bag_label = np.tile(train_bag_gt[idx], len(remove_inst_idx))
        mask_dict[train_bag_idxs[idx]] = remove_inst_idx
        bag_label_list.extend(bag_label), ins_gt_list.extend(preseud_ins_gt)

    removed_Ins_acc = (np.array(bag_label_list)==np.array(ins_gt_list)).mean()
    removed_Ins_purity = (np.array(bag_label_list)!=np.array(ins_gt_list)).mean()
    removed_inst_num = len(ins_gt_list)
    logging.info('===================================================================================')
    logging.info('Removed Instance Accuracy: %.4f' % removed_Ins_acc)
    logging.info('Removed Instance purity: %.4f' % removed_Ins_purity)
    logging.info('Removed Instance num: %d' % removed_inst_num)
    
    boost_majority_label = []
    for idx in range(len(train_bag_gt)):
        mask = np.ones(args.bag_size, dtype=bool)
        mask[mask_dict[train_bag_idxs[idx]]] = False
        ins_label = train_inst_gt[idx][mask]  
        label_num = []
        for c in range(args.classes):
            label_num.append(sum(ins_label==c))
        boost_majority_label.append(np.argmax(np.array(label_num)))
    boost_majority_label= np.array(boost_majority_label)
    
    boost_bag_acc = (np.array(boost_majority_label)==np.array(train_bag_gt)).mean()
    logging.info('Boost bag label Accuracy: %.4f' % boost_bag_acc)
    logging.info('===================================================================================')
    return mask_dict, removed_Ins_acc, removed_Ins_purity, removed_inst_num, boost_bag_acc



def gen_prototype(args, train_inst_gt, train_inst_pred, train_bag_gt, train_bag_pred, train_inst_feat):
    class_ins_feat_dict = {c:[] for c in range(args.classes)}
    train_inst_gt, train_inst_pred, train_inst_feat = train_inst_gt.reshape(-1, args.bag_size), train_inst_pred.reshape(-1, args.bag_size), train_inst_feat.reshape(-1, args.bag_size, 512)
    for idx in range(len(train_bag_gt)):
        if train_bag_pred[[idx]]==train_bag_gt[[idx]]:
            class_ins_feat_dict[train_bag_gt[idx]].extend(train_inst_feat[idx][train_inst_pred[idx]==train_bag_gt[idx]])
    
    # obtain prototype
    prototypes = []
    empty_prototype_class = []
    for c in range(args.classes):
        if len(class_ins_feat_dict[c])==0: # クラスcに推定されたinstがなかった場合には0ベクトルでplace holder
            prototypes.append(np.zeros((512)))
            empty_prototype_class.append(c)
        else:
            prototypes.append(np.mean(np.array((class_ins_feat_dict[c])), axis=0))
    prototypes = np.array(prototypes)
    
    return prototypes, class_ins_feat_dict, empty_prototype_class

