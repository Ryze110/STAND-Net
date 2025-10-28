import torch
import torch.nn as nn

class SimpleMSELoss(nn.Module):
    def __init__(self):
        super(SimpleMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        return self.mse(y_pred, y_true)
