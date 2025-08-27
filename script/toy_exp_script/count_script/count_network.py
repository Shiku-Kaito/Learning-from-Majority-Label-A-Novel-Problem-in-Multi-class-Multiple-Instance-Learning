import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18

class Count(nn.Module):
    def __init__(self, num_class, temper1, temper2):
        super().__init__()

        self.feature_extractor = resnet18(pretrained=False)
        self.feature_extractor.fc = nn.Sequential()

        self.temper1 = temper1
        self.temper2 = temper2

        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        (batch, num_ins, c, w, h) = x.size()
        x = x.reshape(-1, c, w, h)

        y = self.feature_extractor(x)
        y_ins = self.classifier(y)
        y_conf = F.softmax(y_ins ,dim=1)      #softmax with temperature
        y_ins = F.softmax(y_ins / self.temper1 ,dim=1)      #softmax with temperature
        count = y_ins.reshape(batch, num_ins, -1)
        count = count.sum(dim=1)      #カウントする
        count /= count.sum(axis=1, keepdims=True)    #正規化
        y_bag = F.softmax(count / self.temper2 ,dim=1)     #softmax with temperature

        return {"bag": y_bag, "prop": count, "ins": y_ins, "ins_conf": y_conf, "ins_feat": y} #,y, y.reshape(-1,num_ins, 512).mean(dim=1)
