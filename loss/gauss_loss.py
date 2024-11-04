import torch.nn as nn
import torch

class GaussianLoss_Module(nn.Module):
    def __init__(self):
        super(GaussianLoss_Module, self).__init__()
        self.loss_func = nn.GaussianNLLLoss()
    def forward(self,pred_mean,pred_std,gt):
        return self.loss_func(pred_mean,gt,pred_std)