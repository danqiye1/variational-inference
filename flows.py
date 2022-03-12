import torch
import torch.nn as nn
import random
import numpy as np

from pdb import set_trace as bp

# consulted https://github.com/VincentStimper/normalizing-flows/blob/master/normflow/flows/radial.py

class PlanarFlow(nn.Module):
    def __init__(self, z_dim=2):
        super(PlanarFlow, self).__init__()

        self.z_dim = z_dim
        self.h = nn.Tanh()
        self.m = nn.Softplus()

        self.u = nn.init.xavier_normal_(nn.Parameter(torch.empty([z_dim, 1])))
        self.w = nn.init.xavier_normal_(nn.Parameter(torch.empty([z_dim, 1])))
        self.b = nn.init.xavier_normal_(nn.Parameter(torch.empty([1, 1])))

    def forward(self, z, logp):
        a = self.h(torch.mm(z, self.w) + self.b)
        dt = (1 - a ** 2).mm(self.w.T)

        x = self.w.T.mm(self.u)
        m = -1 + self.m(x)
        u_h = self.u + (m - x) * self.w / (self.w.T.mm(self.w))

        logp = logp - torch.log(torch.abs(1 + dt.mm(u_h).squeeze()) + 1e-7)
        z = z + a.mm(u_h.T)

        return z, logp

class RadialFlow(nn.Module):
    '''
    f(z) = z + beta * h(alpha, r) * (z - z_0)
    '''
    def __init__(self, z_dim=2):
        super().__init__()
        
        self.z_dim = z_dim
        self.z0 = nn.init.normal_(nn.Parameter(torch.empty(z_dim)))
        self.alpha = nn.init.normal_(nn.Parameter(torch.empty(1)))
        self.beta = nn.init.normal_(nn.Parameter(torch.empty(1)))
        

    def forward(self, z: torch.Tensor, logp):
        # (z - z0)
        z_sub = z - self.z0
        alpha = torch.abs(self.alpha)
        beta = torch.log(1 + torch.exp(self.beta)) - alpha
        r = torch.linalg.norm(z_sub, dim=1)
        # As shown at the end of section A.2
        h = 1 / (alpha + r)

        # f(z) = z + beta * h(alpha, r) * (z - z_0)
        f_z = z + beta * h.unsqueeze(1) * z_sub
        # Formula as shown at the end of section A.2:  - self.beta * r / (alpha + r) ** 2
        logp -= (((self.z_dim - 1) * torch.log(1 + beta * h) + torch.log(1 + beta * h - beta * r / (alpha + r) ** 2)))
        return f_z, logp
