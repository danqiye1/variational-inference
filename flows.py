import torch
import torch.nn as nn


# class PlanarFlow(nn.Module):
#
#     def __init__(self, z_dim):
#         super(PlanarFlow, self).__init__()
#
#         self.z_dim = z_dim
#         self.h = nn.Tanh()
#         self.m = nn.Softplus()
#
#         self.u = nn.init.xavier_normal(nn.Parameter(torch.empty([z_dim, 1])))
#         self.w = nn.init.xavier_normal(nn.Parameter(torch.empty([z_dim, 1])))
#         self.b = nn.init.xavier_normal(nn.Parameter(torch.empty([1, 1])))
#
#     def forward(self, z, logp):
#         """
#         Given a set of samples z and their respective log probabilities, returns
#         z' = f(z) and log p(z'), as described by the equations in the paper.
#         Sizes should be (L, z_dim) and (L), respectively.
#         Outputs are the same size as the inputs.
#         """
#         a = self.h(torch.mm(z, self.w) + self.b)
#         psi = (1 - a ** 2).mm(self.w.t())  # derivative of tanh(x) is 1 - tanh^2(x)
#
#         # see end of section A.1
#         x = self.w.t().mm(self.u)
#         m = -1 + self.m(x)
#         u_h = self.u + (m - x) * self.w / (self.w.t().mm(self.w))
#
#         logp = logp - torch.abs(torch.log(1 + psi.mm(u_h).squeeze()))
#         z = z + a.mm(u_h.t())
#
#         return z, logp
#
#


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

        logp = logp - torch.log(1 + psi.mm(u_h).squeeze())
        z = z + a.mm(u_h.t())

        return z, logp