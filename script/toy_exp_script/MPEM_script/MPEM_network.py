import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18
# from additive_modules.transmil import TransMIL
# from additive_modules.VIT import VIT
import numpy as np
import random

class MPEM_count_model(nn.Module):
    def __init__(self, num_class, temper1, temper2):
        super().__init__()

        self.feature_extractor = resnet18(pretrained=False)
        self.feature_extractor.fc = nn.Sequential()

        self.temper1 = temper1
        self.temper2 = temper2

        self.classifier = nn.Linear(512, num_class)

    def forward(self, x, mask_idx, bag_idxs):
        (batch, num_ins, c, w, h) = x.size()
        x = x.reshape(-1, c, w, h)
        
        y = self.feature_extractor(x)
        y_ins = self.classifier(y)
        y_conf = F.softmax(y_ins ,dim=1)      #softmax with temperature
        y_ins = F.softmax(y_ins / self.temper1 ,dim=1)      #softmax with temperature

        counts = y_ins.reshape(batch, num_ins, -1)
        count_list, y_bags = [], []
        for idx in range(batch):            # loop at all bag
            if bag_idxs==None:
                count = counts[idx]
            else:
                mask = np.ones(num_ins, dtype=bool)
                mask[mask_idx[int(bag_idxs[idx])]] = False
                count = counts[idx][mask]
            count = count.sum(dim=0)      #カウントする
            count /= count.sum(axis=0, keepdims=True)    #正規化
            count_list.append(count)
            y_bag = F.softmax(count / self.temper2 ,dim=0)     #softmax with temperature    
            y_bags.append(y_bag)
            
        y_bags, count_list = torch.stack((y_bags)), torch.stack((count_list))
        return {"bag": y_bags, "prop": count_list, "ins": y_ins, "ins_conf": y_conf, "ins_feat":y} #,y, y.reshape(-1,num_ins, 512).mean(dim=1)
    
