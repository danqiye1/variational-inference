import torch
import torch.nn as nn
import random

class PlanarFlow(nn.Module):
    """
    Planar flow transformation, described in equations (10) - (12), (21) - (23) in https://arxiv.org/pdf/1505.05770.pdf
    We use tanh activation (h) function.
    """

    def __init__(self, z_dim=2):
        super(PlanarFlow, self).__init__()

        self.z_dim = z_dim
        self.h = nn.Tanh()
        self.m = nn.Softplus()

        self.u = nn.init.xavier_normal_(nn.Parameter(torch.empty([z_dim, 1])))
        self.w = nn.init.xavier_normal_(nn.Parameter(torch.empty([z_dim, 1])))
        self.b = nn.init.xavier_normal_(nn.Parameter(torch.empty([1, 1])))

    def forward(self, z, logp):
        """
        Given a set of samples z and their respective log probabilities, returns
        z' = f(z) and log p(z'), as described by the equations in the paper.
        Sizes should be (L, z_dim) and (L), respectively.
        Outputs are the same size as the inputs.
        """

        a = self.h(torch.mm(z, self.w) + self.b)
        psi = (1 - a ** 2).mm(self.w.t())  # derivative of tanh(x) is 1 - tanh^2(x)

        # see end of section A.1
        x = self.w.t().mm(self.u)
        m = -1 + self.m(x)
        u_h = self.u + (m - x) * self.w / (self.w.t().mm(self.w))

        logp = logp - torch.log(torch.abs(1 + psi.mm(u_h).squeeze()) + 1e-7)
        z = z + a.mm(u_h.t())

        return z, logp

class RadialFlow(nn.Module):
    def __init__(self, z_dim=2):
        super(RadialFlow, self).__init__()

        self.z_dim = z_dim
        self.m = nn.Softplus()

        #?
        self.beta = nn.init.xavier_normal_(nn.Parameter(torch.empty([1, 1])))
        self.alpha = nn.init.xavier_normal_(nn.Parameter(torch.empty([1, 1])))
        self.z0 = nn.init.uniform_(nn.Parameter(torch.empty([z_dim, 1])), a = 1e-1, b = 1)
        
    def forward(self, z, logp):
        diff = z - self.z0
        r = abs(diff)
        h = 1/(self.alpha + r) #dim of one single z
        beta = -self.alpha + self.m(self.beta)
        z = z + beta * torch.mm(h, diff)

        first = 1 + beta*self.h
        h_ = self.h ##to-do, mm or *?
        second = beta * torch.mm(h_, r)
        det = torch.mm(first, (first + second)) ##to-do, what does the d-1 mean?
        logp = logp - torch.log(det)
        return z, logp

        #to-do: test
