import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
import numpy as np

class FlowVAE(nn.Module):
    def __init__(self, img_size, dim_h, dim_z, flows, decoder):
        """
        normalizing flow model
        :param img_size: shape of the image
        :param dim_h: dimension of hidden states
        :param dim_z: dimension of latent variable
        :param flows: Flows to transform output of base encoder
        """
        super().__init__()
        self.dim_z = dim_z
        self.img_size = img_size
        self.prior = Normal(0., 1.)
        self.encoder = nn.Sequential(nn.Linear(img_size[0]*img_size[1]*img_size[2], 2*dim_h), nn.ReLU(True), nn.Linear(2*dim_h, dim_h), nn.ReLU(True))
        self.mu = nn.Linear(dim_h, dim_z)
        self.var = nn.Linear(dim_h, dim_z)

        self.flows = nn.ModuleList(flows)
        self.decoder = decoder

    def forward(self, x):
        """
        Takes data batch, samples num_samples for each data point from base distribution
        :param x: data
        :return: generated image and kl divergence
        """
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.var(x)

        # reparameterization
        sigma = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(mu) * sigma
        q = Normal(mu, sigma)
        logq = q.log_prob(z)

        logp = -0.5 * torch.sum(2 * torch.log(sigma) + np.log(2 * np.pi) + ((z - mu) / sigma) ** 2, dim=1)
        for flow in self.flows:
            z, logp = flow(z, logp)

        kl = - torch.sum(self.prior.log_prob(z), dim=-1) + torch.sum(logq, dim=-1) - logp
        # likelihood
        likelihood = self.decoder(z)
        return likelihood, kl

    def sample_img(self):
        with torch.no_grad():
            z = torch.randn(1, self.dim_z)
            # The attribute probs is a little weird as we have to use it in Bernoulli. We desgin the variable of same name for LogitNormal
            out = self.decoder(z).probs
            out = out.reshape(-1, self.img_size[0], self.img_size[1], self.img_size[2])
        return out


class BernoulliDecoder(nn.Module):

    def __init__(self, img_size, dim_z, dim_h) -> None:
        super().__init__()
        self.decoder_nn = nn.Sequential(nn.Linear(dim_z, dim_h), nn.ReLU(True), nn.Linear(dim_h, 2*dim_h), nn.ReLU(True), nn.Linear(2*dim_h, img_size[0]*img_size[1]*img_size[2]))

    def forward(self, z):
        return Bernoulli(torch.sigmoid(self.decoder_nn(z)))


class LogitNormalDecoder(nn.Module):

    def __init__(self, img_size, dim_z, dim_h) -> None:
        super().__init__()
        self.decoder_nn = nn.Sequential(nn.Linear(dim_z, dim_h), nn.ReLU(True), nn.Linear(dim_h, 2*dim_h), nn.ReLU(True))
        self.mu = nn.Linear(2*dim_h, img_size[0]*img_size[1]*img_size[2])
        self.var = nn.Linear(2*dim_h, img_size[0]*img_size[1]*img_size[2])

    def forward(self, z):
        z = self.decoder_nn(z)
        mu = self.mu(z)
        log_var = self.var(z)
        sigma = torch.exp(0.5 * log_var)
        self.normal = Normal(mu, sigma)
        self.probs = torch.exp(self.normal.mean) / (1 + torch.exp(self.normal.mean))  # weird name
        return self

    def log_prob(self, x):
        logit_x = torch.log(x / (1 - x) + 1e-5)
        log_norm_prob = self.normal.log_prob(logit_x)
        return log_norm_prob - torch.log(x*(1 - x))
