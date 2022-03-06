import torch
import torch.nn as nn
import random
import numpy as np

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
    def __init__(self, z_dim=2):
        super(RadialFlow, self).__init__()
        self.d_cpu = torch.prod(torch.tensor(z_dim))
        self.register_buffer('d', self.d_cpu)
        self.beta = nn.Parameter(torch.empty(1))
        lim = 1.0 / np.prod(z_dim)
        nn.init.uniform_(self.beta, -lim - 1.0, lim - 1.0)
        self.alpha = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.alpha, -lim, lim)

        self.z_0 = nn.Parameter(torch.randn(z_dim)[None])

    def forward(self, z):
        beta = torch.log(1 + torch.exp(self.beta)) - torch.abs(self.alpha)
        dz = z - self.z_0
        r = torch.norm(dz, dim=list(range(1, self.z_0.dim())))
        h_arr = beta / (torch.abs(self.alpha) + r)
        h_arr_ = - beta * r / (torch.abs(self.alpha) + r) ** 2
        z_ = z + h_arr.unsqueeze(1) * dz
        log_det = (self.d - 1) * torch.log(1 + h_arr) + torch.log(1 + h_arr + h_arr_)
        return z_, log_det