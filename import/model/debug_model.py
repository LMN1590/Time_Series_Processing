import torch.nn as nn
import torch

class DebugModel(nn.Module):
    def __init__(
        self,
        input_size:int,
        hidden_size:int,
        output_size:int
    ):
        super(DebugModel, self).__init__()
        
        self.layer_1 = nn.Linear(
            input_size,hidden_size
        )
        self.layer_2 = nn.Linear(
            hidden_size,output_size
        )
        self.relu = nn.ReLU()
    
    def forward(self, x:torch.Tensor):
        B,T,I = x.shape
        x = x.reshape((B,-1))
        
        x = self.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x