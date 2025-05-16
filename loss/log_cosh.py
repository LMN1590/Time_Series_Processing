import torch.nn as nn
import torch

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - torch.log(torch.tensor(2.0))
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()
    def forward(self,pred_mean,pred_std,gt):
        return log_cosh_loss(pred_mean, gt)

    
if __name__ == "__main__":
    loss = LogCoshLoss()
    
    sample_1 = torch.randn(10)
    sample_2 = torch.randn(10)
    print(loss(sample_1,sample_1,sample_2))



        