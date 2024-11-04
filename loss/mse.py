from torch.nn import MSELoss
import torch.nn as nn

class MSE_Module(nn.Module):
    def __init__(self):
        super(MSE_Module, self).__init__()
        self.loss_func = MSELoss()
    def forward(self,pred_mean,pred_std,gt):
        return self.loss_func(pred_mean,gt)
    