import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from tqdm import tqdm

def plot_function(u_func, axis):
    side = np.linspace(-4, 4, 1000)
    X, Y = np.meshgrid(side, side)
    Z = np.concatenate([np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))], 1)
    neg_logp = u_func(torch.Tensor(Z)).numpy()
    neg_logp = np.reshape(neg_logp, X.shape)
    p = np.exp(-neg_logp)
    p /= np.sum(p)
    Y = -Y
    axis.pcolormesh(X, Y, p, shading='auto')


def plot_model_hist(model, ax=plt, size=(-5, 5), num_side=500):
    side = np.linspace(size[0], size[1], num_side)
    X, Y = np.meshgrid(side, side)
    counts = np.zeros(X.shape)
    p = np.zeros(X.shape)

    batch_size = 100000
    z_dim = 2
    for i in tqdm(range(10), desc='Sampling'):

        prior = MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        z = prior.rsample([batch_size])
        logq = prior.log_prob(z)

        for flow in model:
            z, logq = flow(z, logq)
        q_k = torch.exp(logq)
        z = (z - size[0]) * num_side / (size[1] - size[0])
        for l in range(batch_size):
            x, y = int(z[l, 1]), int(z[l, 0])
            if 0 <= x and x < num_side and 0 <= y and y < num_side:
                counts[x, y] += 1
                p[x, y] += q_k[l]

    counts = np.maximum(counts, np.ones(counts.shape), shading='auto')
    p /= counts
    p /= np.sum(p)
    Y = -Y
    ax.pcolormesh(X, Y, p)
