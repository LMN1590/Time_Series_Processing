from torch.linalg import norm
import torch.nn as nn

class StdReg_Module(nn.Module):
    def __init__(self):
        super(StdReg_Module, self).__init__()
    def forward(self,pred_mean,pred_std,gt):
        return norm(pred_std)
    