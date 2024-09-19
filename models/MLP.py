"""
Description: 
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2024-09-19 22:01:25
"""

import torch
import torch.nn as nn


# define the model
class ScoringModel(nn.Module):
    def __init__(
        self,
    ):
        super(ScoringModel, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(-9.0))
        self.gamma = nn.Parameter(torch.tensor(0.6))

    def forward(self, m, f, tani, cors):
        # input: m,f,tani
        # output: s

        sig = torch.sigmoid(-self.beta * (m * tani - self.gamma))
        s = self.alpha * f + (1 - self.alpha) * torch.sum(torch.abs(cors) * sig, dim=1)
        # s = self.alpha * f + (1 - self.alpha) * torch.sum(sig)

        return s
