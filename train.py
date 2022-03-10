import os
import argparse
import torch
from flows import PlanarFlow
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
from NormVAE import FlowVAE, BernoulliDecoder, LogitNormalDecoder


def prepare_dataset(dataset_name, batch_size, train=True):
    data = {
        'mnist': datasets.MNIST('datasets', train=train, download=True, transform=transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).type(x.type())])),
        'cifar10': datasets.CIFAR10(
            'datasets', train=train, download=True, transform=transforms.Compose(
                #[transforms.RandomCrop([8, 8]), transforms.ToTensor(), lambda x: (0.999 - 0.001) * x + 0.001]
                [transforms.ToTensor(), lambda x: (0.999 - 0.001) * x + 0.001]
            )
        )
    }
    dataloader = torch.utils.data.DataLoader(data[dataset_name], batch_size=batch_size, shuffle=train)
    return dataloader


def train(model, dataset_name, beta=1, device='cpu', num_epoch=10):
    dataloader = prepare_dataset(dataset_name, batch_size)
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    for epoch in range(num_epoch):
        t = tqdm(enumerate(dataloader), total=len(dataloader))
        epoch_loss = []
        for i, (x, _) in t:
            # with torch.autograd.set_detect_anomaly(True):
            x = x.to(device)
            optimizer.zero_grad()
            x = x.reshape([-1, x.shape[1]*x.shape[2]*x.shape[3]])
            likelihood, kl = model(x)
            recon_loss_ = -likelihood.log_prob(x).sum() / len(x)
            kl_loss_ = kl.mean()
            loss = recon_loss_ + beta * kl_loss_
            loss.backward()
            epoch_loss.append(loss.item())
            min_ln_loss_ = min([val.sum() for val in -likelihood.log_prob(x)])
            max_ln_loss_ = max([val.sum() for val in -likelihood.log_prob(x)])
            min_kl_loss_ = kl.min()
            max_kl_loss_ = kl.max()
            t.set_description("Epoch %d, Recon loss %.3f (min-max %.3f %.3f), KL %.3f (min-max %.3f %.3f)" %
                              (epoch, recon_loss_, min_ln_loss_, max_ln_loss_,
                               kl_loss_, min_kl_loss_, max_kl_loss_))
            optimizer.step()

        if epoch and epoch % save_frequency == 0:
            model.save_model(save_path=save_path, epoch=epoch)
    return model

def train_energy(U, model, z_dim=2, batch_size=512, iteration=5000):
    """ Function for training on energy densities """

    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    t = tqdm(range(iteration))
    for it in t:
        optimizer.zero_grad()

        # prior
        prior = MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        z = prior.rsample([batch_size])
        logq = prior.log_prob(z)

        for flow in model:
            z, logq = flow(z, logq)

        U_z = U(z)
        loss = torch.mean(U_z + logq)
        loss.backward()
        t.set_description("Loss %f" % loss)

        optimizer.step()
    return model


def test(model, dataset_name, num_flows, batch_size, epoch):
    dataloader = prepare_dataset(dataset_name, batch_size, train=False)
    model.load_model(f"checkpoint/{dataset_name}/{num_flows}", epoch)
    model.eval()
    t = tqdm(enumerate(dataloader), total=len(dataloader))
    recon_loss = []
    kl_loss = []
    for i, (x, _) in t:
        x = x.reshape([-1, x.shape[1]*x.shape[2]*x.shape[3]])
        with torch.no_grad():
            likelihood, kl = model(x)
        recon_loss_ = -likelihood.log_prob(x).sum() / len(x)
        kl_loss_ = kl.mean()
        recon_loss.append(recon_loss_)
        kl_loss.append(kl_loss_)
    return np.array(recon_loss).mean(), np.array(kl_loss).mean()

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description="Normalizing Flow VAE")
    parser.add_argument("--dataset", '-d', dest="dataset_name", default="mnist", type=str)
    parser.add_argument("--numflows", '-k', dest="num_flows", default=10, type=int)
    parser.add_argument("--epochs", '-e', dest="num_epoch", default=10, type=int)
    parser.add_argument("--beta", '-b', dest="beta", default=1, type=float)
    args = parser.parse_args()

    # Input checking
    assert args.dataset_name in ("mnist", "cifar10"), f"{args.dataset_name} is not valid dataset. Must be mnist or cifar10."

    # Static hyperparameters
    batch_size = 256
    dim_z = 40
    dim_h = 256
    save_frequency = 2
    beta = 1  # assign beta

    # Derived hyperparameters
    img_size = [1, 28, 28] if args.dataset_name == 'mnist' else [3, 32, 32]
    save_path = os.path.join('./checkpoint', f'{args.dataset_name}/{args.num_flows}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decoder = BernoulliDecoder(img_size, dim_z, dim_h) if args.dataset_name == "mnist" else LogitNormalDecoder(img_size, dim_z, dim_h)

    # assign flows if None, use standard VAE
    flows = [PlanarFlow(dim_z) for _ in range(args.num_flows)]

    model = FlowVAE(img_size, dim_h, dim_z, decoder, flows)
    train(model, args.dataset_name, beta, device, num_epoch=args.num_epoch)
