"""
Description: 
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2024-11-17 15:09:02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Metfusion_Our(nn.Module):
    def __init__(
        self,
    ):
        super(Metfusion_Our, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(-9.0))
        self.gamma = nn.Parameter(torch.tensor(0.6))
        self.lamda = nn.Parameter(torch.tensor(0.5))

    def forward(self, m, f, tani, cors):
        # input: m: (1, ), f: (N, ), tani: (N, 1), cors: (1, )
        # output: s: (N, )

        # situation of correlation outside the sigmoid function
        sig = torch.sigmoid(-self.beta * (m * tani - self.gamma))  # 336,1
        s = (
            self.alpha * f
            + (1 - self.alpha) * torch.sum(sig, dim=1)
            + self.lamda * torch.sum(torch.abs(cors) * sig, dim=1)
        )

        return s


class Metfusion_Our_v2(nn.Module):

    def __init__(
        self,
    ):
        super(Metfusion_Our_v2, self).__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, m, f, tani, cors):
        # input: m: (1, ), f: (N, ), tani: (N, 1), cors: (1, )
        # output: s: (N, )

        m = m.repeat(tani.size(0), 1)
        cors = cors.repeat(tani.size(0), 1)
        s_list = []
        for i in range(m.size(1)):
            x = torch.cat(
                [
                    m[:, i].unsqueeze(1),
                    f.unsqueeze(1),
                    tani[:, i].unsqueeze(1),
                    cors[:, i].unsqueeze(1),
                ],
                dim=1,
            )
            s = self.fc(x)
            s_list.append(s)
        s = torch.sum(torch.cat(s_list, dim=1), dim=1)
        return s


class Metfusion_Our_v3(nn.Module):

    def __init__(self, hidden_size):
        super(Metfusion_Our_v3, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, m, f, tani, cors):
        # input: m: (1, ), f: (N, ), tani: (N, 1), cors: (1, )
        # output: s: (N, )

        m = m.repeat(tani.size(0), 1)
        cors = cors.repeat(tani.size(0), 1)
        s_list = []
        for i in range(m.size(1)):
            x = torch.cat(
                [
                    m[:, i].unsqueeze(1),
                    f.unsqueeze(1),
                    tani[:, i].unsqueeze(1),
                    cors[:, i].unsqueeze(1),
                ],
                dim=1,
            )
            s = self.fc2(F.relu(self.fc1(x)))

            s_list.append(s)
        s = torch.sum(torch.cat(s_list, dim=1), dim=1)
        return s
