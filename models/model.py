"""
Description: 
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2024-09-27 13:20:34
"""

import torch
import torch.nn as nn


# define the model
class Metfusion(nn.Module):
    def __init__(
        self,
    ):
        super(Metfusion, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(-9.0))
        self.gamma = nn.Parameter(torch.tensor(0.6))

    def forward(self, m, f, tani, cors):
        # input: m,f,tani
        # output: s

        sig = torch.sigmoid(-self.beta * (m * tani - self.gamma))
        s = self.alpha * f + (1 - self.alpha) * torch.sum(sig, dim=1)

        return s


class Metfusion_Cor_Inside(nn.Module):

    def __init__(
        self,
    ):
        super(Metfusion_Cor_Inside, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(-9.0))
        self.gamma = nn.Parameter(torch.tensor(0.6))

    def forward(self, m, f, tani, cors):
        # input: m,f,tani
        # output: s

        sig = torch.sigmoid(-self.beta * (torch.abs(cors) * m * tani - self.gamma))
        s = self.alpha * f + (1 - self.alpha) * torch.sum(sig, dim=1)

        return s


class Metfusion_Cor_Outside(nn.Module):
    def __init__(
        self,
    ):
        super(Metfusion_Cor_Outside, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(-9.0))
        self.gamma = nn.Parameter(torch.tensor(0.6))

    def forward(self, m, f, tani, cors):
        # input: m,f,tani
        # output: s

        # situation of correlation outside the sigmoid function
        sig = torch.sigmoid(-self.beta * (m * tani - self.gamma))
        s = self.alpha * f + (1 - self.alpha) * torch.sum(torch.abs(cors) * sig, dim=1)

        return s
