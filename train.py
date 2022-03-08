import os
import torch
from flows import PlanarFlow
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from NormVAE import FlowVAE, BernoulliDecoder, LogitNormalDecoder


def prepare_dataset(dataset_name, batch_size, train=True):
    data = {
        'mnist': datasets.MNIST('datasets', train=train, download=True, transform=transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).type(x.type())])),
        'cifar10': datasets.CIFAR10('datasets', train=train, download=True, transform=transforms.Compose(
            # ToTensor will
        [transforms.RandomCrop([8, 8]), transforms.ToTensor(), lambda x: (0.999 - 0.001) * x + 0.001]))
    }
    dataloader = torch.utils.data.DataLoader(data[dataset_name], batch_size=batch_size, shuffle=train)
    return dataloader


def train(model, dataset_name, beta=1, device='cpu'):
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
            t.set_description("Recon loss %f (min-max %f %f), KL %f (min-max %f %f)" %
                              (recon_loss_, min_ln_loss_, max_ln_loss_,
                               kl_loss_, min_kl_loss_, max_kl_loss_))
            optimizer.step()

        if epoch and epoch % save_frequency == 0:
            model.save_model(save_path=save_path, epoch=epoch)
    return model


def test(model, dataset_name, load_epoch):
    dataloader = prepare_dataset(dataset_name, batch_size, train=False)
    model.load_model(save_path=save_path, epoch=load_epoch)
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
    num_epoch = 100
    dataset_name = 'mnist'  # 'cifar10'
    img_size = [1, 28, 28] if dataset_name == 'mnist' else [3, 8, 8]
    batch_size = 256
    num_flow = 10
    dim_z = 40
    dim_h = 256
    save_path = os.path.join('./checkpoint', dataset_name)
    save_frequency = 2
    beta = 1  # assign beta
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dataset_name == 'mnist':
        decoder = BernoulliDecoder(img_size, dim_z, dim_h)
    elif dataset_name == 'cifar10':
        decoder = LogitNormalDecoder(img_size, dim_z, dim_h)

    # assign flows if None, use standard VAE
    flows = [PlanarFlow(dim_z) for _ in range(num_flow)]

    model = FlowVAE(img_size, dim_h, dim_z, decoder)
    train(model, dataset_name, beta, device)
