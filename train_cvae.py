# a simple variational autoencoder model using pytorch for a dataset of 19 input dims and a latent space of 3 dims
import torch
import torch.nn as nn
import torch.utils.data as tud
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
from torch.nn import functional as F
matplotlib.use('Agg') 
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import argparse

import os
import glob
from pathlib import Path


def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

class CNNEncoder(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        self.image_shape = image_shape
        self.conv = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, *image_shape)).shape[1]
        self.fc = nn.Linear(flat_dim, 128)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, *self.image_shape)
        x = self.conv(x)
        x = self.fc(x)
        return x

# cvae
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, dataset, class_size, image_shape=None):
        super().__init__()
        self.input_dim = input_dim
        self.image_shape = image_shape
        self.is_image = dataset == 'scenairo'
        if dataset == 'mnist':  # MNIST
            self.cnn = CNNEncoder((1, 28, 28))
            encoder_output_dim = 128
        elif self.is_image:
            self.cnn = CNNEncoder(image_shape)
            encoder_output_dim = 128
        else:  # ADIMA
            self.cnn = None
            encoder_output_dim = input_dim
        self.calss_size = class_size

        # encode
        self.encoder = nn.Sequential(
            nn.Linear(encoder_output_dim + class_size, 64),
            nn.ReLU()
        )
        self.fc21 = nn.Linear(64, latent_dim)
        self.fc22 = nn.Linear(64, latent_dim)
    
        # decode
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + class_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

        self.sigmoid = nn.Sigmoid()
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x, c):
        # encode
        if self.cnn is not None:
            x = self.cnn(x)
        enc_inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.encoder(enc_inputs)
        mu = self.fc21(h1)
        log_var = self.fc22(h1)

        z = self.reparameterize(mu, log_var) # sample

        # decode
        dec_inputs = torch.cat([z, c], 1) # (bs, latent_dim+class_size)
        recon = self.decoder(dec_inputs)
        if self.is_image:
            recon = self.sigmoid(recon)
            recon = recon.view(-1, *self.image_shape)

        return recon, mu, log_var

def load_data(data_csv, dataset='adima', single_class=None):
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        X = mnist_train.data.reshape(-1, 784).float() / 255.0
        X = X.numpy()  # Convert tensor to numpy for train_test_split
        y = mnist_train.targets.numpy()  # Convert tensor to numpy
    else:
        data = data_csv
        data= pd.read_csv(data)
        # fiter to a single class
        if single_class is not None:
            data = data[data['Spannung'] == single_class]
        X = data.drop(columns=['Spannung'])
        #features = features.iloc[:,:]
        y = data['Spannung'] # labels
        
    # print the first and last 5 elements of y
    print(f"labels head(): {y[0:5]}")

    # split train/test
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

    # handle both numpy arrays and dataframes
    if isinstance(X_train, np.ndarray):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    else:
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    
    if isinstance(X_valid, np.ndarray):
        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    else:
        X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    
    return X_train_tensor, X_valid_tensor, X, X_train, y_train

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #return BCE + KLD
    return recon_loss + 2*KLD

# training loop for VAE
def learn(epochs, train_loader, vae, loss_fn, optimizer):
    outputs = []
    vae_losses = []
    for epoch in range(epochs):
        epoch_loss = 0

        for batch in tqdm(train_loader):
            if len(batch) == 3:
                data, labels, rot_target = batch
            else:
                data, labels = batch
                rot_target = None
            if not vae.is_image:
                # data from the train loader is torch.Size([16, 1, 28, 28])
                data = data.view(data.size(0), -1)  # flatten it to [batch_size, 784]
            labels = one_hot(labels, vae.calss_size)
            recon_batch, mu, log_var = vae(data, labels)
            optimizer.zero_grad()
            batch_loss = loss_function(recon_batch, data, mu, log_var)

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        outputs.append((epoch, data, recon_batch))
        vae_losses.append(avg_loss)

    return outputs, vae_losses

def plot(ae_losses):
    # plot loss
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(ae_losses)
    plt.savefig('vae_loss.png')
    # plt.show()

def main():
    root = Path(__file__).parent.parent.resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vae_')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--data_csv', default=root/'data/adima/training/fixed_current_training_data.csv')
    parser.add_argument('--model_dir', default='./runs')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=12)
    parser.add_argument('--data_root', default='DATA/ScenAIro')
    args = parser.parse_args()

    if args.dataset == None:
        raise ValueError("enter --dataset 'mnist' or 'senairo' ")
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        input_dim = 784
        class_size = 10
        image_shape = None
    elif args.dataset == 'scenairo':
        transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(Path(args.data_root) / 'train', transform=transform)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        image_shape = (3, 72, 128)
        input_dim = image_shape[0] * image_shape[1] * image_shape[2]
        class_size = len(dataset.classes)
        print("Class mapping:", dataset.class_to_idx)
    else:
        # to train with only green LEDs
        green_leds = args.data_csv
        X_train_tensor, X_valid_tensor, X, _, Y_train = load_data(args.data_csv, args.dataset, single_class=None)
        X_train_tensor_min = X_train_tensor.min()
        X_train_tensor_max = X_train_tensor.max()
        X_train_tensor = (X_train_tensor - X_train_tensor_min) / (X_train_tensor_max - X_train_tensor_min + 1e-8)
        print("Y_train: ", Y_train)
        label_digits = {"blue": 0, "green": 1, "red": 2, "yellow": 3}
        y_train_mapped = Y_train.map(label_digits).astype(int)
        dataset = TensorDataset(X_train_tensor, torch.tensor(y_train_mapped.values))  # include the labels in the torch dataloader
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(X_valid_tensor, batch_size=16, shuffle=False)
        input_dim = 19
        class_size = 4
        image_shape = None

    # training params
    cvae_model = CVAE(input_dim=input_dim, latent_dim=args.latent_dim, dataset=args.dataset, class_size=class_size, image_shape=image_shape)
    cvae_model = cvae_model.train() # set to training mode
    loss_fn = nn.GaussianNLLLoss(reduction='sum', eps=1e-6)  # Learn per-feature variance during reconstruction
    optimizer = optim.Adam(cvae_model.parameters(), lr=1e-3)
    # train
    outputs, vae_losses = learn(args.epochs, train_loader, cvae_model, loss_fn, optimizer)
    plot(vae_losses)
    # save model
    log_dir = f"{args.model_dir}/{args.model_name}"
    os.makedirs(log_dir, exist_ok=True)
    # torch.save(AutoEnc.state_dict(), f"ae-runs/{args.model_name}/{args.model_name}.pth")
    n = max([int(Path(f).stem.split('_')[1]) for f in glob.glob(f'{log_dir}/vae_*.pth')] + [0]) + 1
    torch.save(cvae_model.state_dict(), f"{args.model_dir}/{args.model_name}/vae_{n}.pth")

    print(f"Model saved to {args.model_dir}/{args.model_name}/vae_{n}.pth")    

if __name__ == '__main__':
 main()
