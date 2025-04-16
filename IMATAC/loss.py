import torch
from torch import nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target, weight):
        # Ensure dimensions match for broadcasting
        target = target.unsqueeze(1)
        weight = weight.unsqueeze(1)
        input = input.unsqueeze(1)

        # Pad input to match target shape
        input = F.pad(input, (0, target.shape[2] - input.shape[2]))
        return ((input - target) ** 2) * weight


class UncertainLoss(nn.Module):
    def __init__(self, num, init_method='normal', **init_params):
        super(UncertainLoss, self).__init__()
        self.num = num
        self.sigma = nn.Parameter(torch.randn(num))
        self.init_sigma(init_method, **init_params)

    def init_sigma(self, method='constant', value=1.0, mean=3.0, std=1.0):
        with torch.no_grad():
            if method == 'constant':
                self.sigma.fill_(value)
            elif method == 'random':
                self.sigma.uniform_(0, 1)  # Random values between 0 and 1
            elif method == 'normal':
                self.sigma.normal_(mean, std)  # Normal distribution
            else:
                raise ValueError(f"Unknown initialization method: {method}")

    def forward(self, *inputs):
        loss = 0
        for i in range(self.num):
            loss += inputs[i] / (2 * self.sigma[i] ** 2)
        loss += torch.log(torch.prod(self.sigma.pow(2)))
        return loss