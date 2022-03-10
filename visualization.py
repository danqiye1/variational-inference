import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from train import train_energy
from flows import PlanarFlow, RadialFlow

from pdb import set_trace as bp

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


def plot_model_hist(model, ax=plt, size=(-5, 5), num_side=500, z_dim=2, batch_size=10000):
    side = np.linspace(size[0], size[1], num_side)
    X, Y = np.meshgrid(side, side)
    counts = np.zeros(X.shape)
    p = np.zeros(X.shape)

    # batch_size = 100000
    # z_dim = 2
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

    counts = np.maximum(counts, np.ones(counts.shape))
    p /= counts
    p /= np.sum(p)
    Y = -Y
    ax.pcolormesh(X, Y, p)

if __name__ == "__main__":

    num_flow = 32
    z_dim = 2
    iteration = 5000
    batch_size = 512

    w1 = lambda x: torch.sin(2 * np.pi * x[:, 0] / 4)
    w2 = lambda x: 3 * torch.exp(-0.5 * ((x[:, 0] - 1) / 0.6) ** 2)
    w3 = lambda x: 3 * torch.sigmoid((x[:, 0] - 1) / 0.3)

    def u1(z):
        """
        :param z: The first dimension means batch
        :return: value of the function
        """
        return 0.5 * ((torch.norm(z, 2, dim=1) - 2) / 0.4) ** 2 \
            - torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
                        + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2))

    def u2(z):
        return 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2

    def u3(z):
        return -torch.log(torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2) +
                        torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2))

    def u4(z):
        return -torch.log(torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2) +
                        torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2))


    # For speed of testing and debugging we just test one energy function
    planar_model_list = []
    model = nn.ModuleList([RadialFlow(z_dim) for i in range(num_flow)])
    planar_model_list.append(train_energy(u1, model, z_dim, batch_size, iteration))

    fig, axes = plt.subplots(2, 2)
    for model, ax in zip(planar_model_list, axes.flatten()):
        plot_model_hist(model, ax, z_dim=z_dim, batch_size=batch_size)
        ax.axis('off')

    fig.tight_layout()
    plt.show()