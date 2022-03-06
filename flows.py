import torch
import torch.nn as nn
import random

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

        self.z_dim = z_dim
        self.m = nn.Softplus()

        self.beta = nn.init.xavier_normal_(nn.Parameter(torch.empty([1, 1])))
        self.alpha = nn.init.xavier_normal_(nn.Parameter(torch.empty([1, 1])))
    
    def getF(self, z):
        self.z0 = torch.empty(z.shape)
        self.z0 = nn.init.uniform_(self.z0, a = 1, b = 2)
        diff = z - self.z0
        self.r = abs(diff)
        self.h = 1/(self.alpha + self.r)
        self.newbeta = -self.alpha + self.m(self.beta)
        z = z + self.newbeta * torch.mul(self.h, diff)
        return z
    
    def getH(self, z):
        return 1/(self.alpha.detach().numpy() + abs(z - self.z0.detach().numpy()))
        
    def forward(self, z, logp):
        newZ = self.getF(z)
        
        bh1 = self.newbeta * self.h + 1
        
        hf = elementwise_grad(self.getH)
        h_ =  torch.from_numpy(hf(z.detach().numpy())).float()

        det = torch.mm(bh1 ** (self.z_dim - 1), \
                        (bh1 + self.newbeta * h_ * self.r).T)
        
        logp = logp - torch.log(det + 1e-7)
        return newZ, logp