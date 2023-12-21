import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn
import pickle
import random
import numpy as np
from torchvision.datasets import MNIST
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150


class ColoredMNIST(MNIST):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, train=train, download=download, transform=transform,
                                           target_transform=target_transform)

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        img = img.numpy()
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = np.stack((img, img, img), axis=2)
        img = np.where(img > 0, color, img)
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
colored_mnist_train = ColoredMNIST(root='./data', train=True, download=True, transform=transform)
colored_mnist_test = ColoredMNIST(root='./data', train=False, download=True)
train_loader = torch.utils.data.DataLoader(dataset=colored_mnist_train, batch_size=120, shuffle=True, num_workers=4)


# Continuous variational autoencoder

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(2352, 512)
        self.linear2 = nn.Linear(512, 256)
        self.to_mean_logvar = nn.Linear(256, 2 * latent_dims)

    def reparametrization_trick(self, mu, log_var):
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu, log_var = torch.split(self.to_mean_logvar(x), 2, dim=-1)
        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return self.reparametrization_trick(mu, log_var)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 2352)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = torch.sigmoid(self.linear3(z))
        z = z.reshape(-1, 3, 28, 28)
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_continuous(data, epochs=20):
    z_dim = 2
    model = VariationalAutoencoder(z_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        running_loss = 0.0
        print(f'Epoch {epoch + 1}')
        print("-" * 40)
        for x, y in data:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = F.binary_cross_entropy(x_hat, x, reduction='sum') + model.encoder.kl
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch loss: {running_loss} \n")

    with open('vae_continuous.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model


def plot_latent(autoencoder, data):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')

    plt.colorbar()
    plt.title("Latent visualization")


def plot_digits_continuous(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    img = []
    for i, z2 in enumerate(np.linspace(r1[1], r1[0], n)):
        for j, z1 in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[z1, z2]]).to(device)
            x_hat = autoencoder.decoder(z)
            img.append(x_hat)

    img = torch.cat(img)
    img = torchvision.utils.make_grid(img, nrow=n).permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(img, extent=[*r0, *r1])
    plt.title(f"z0 : {r0}, z1: {r1}", fontweight="bold")
    plt.show()

def main_continuous():
    model = train_continuous(train_loader, epochs=30)
    plot_latent(model, train_loader)

    plot_digits_continuous(model, r0=(2, 3), r1=(3, 4), n=2)
    plt.show()
    plot_digits_continuous(model, r0=(3.4, 3.7), r1=(-2.3, -2), n=2)
    plt.show()
    plot_digits_continuous(model, r0=(0, 0.05), r1=(-4, -3.95), n=2)
    plt.show()
    plot_digits_continuous(model, r0=(-4, -3.95), r1=(3.95, 4), n=2)
    plt.show()


# Discrete variational autoencoder
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-20):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps).to(device)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    bs, N, K = logits.size()
    y_soft = gumbel_softmax_sample(logits.view(bs * N, K), tau=tau, eps=eps)

    if hard:
        k = torch.argmax(y_soft, dim=-1)
        y_hard = F.one_hot(k, num_classes=K)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y.reshape(bs, N * K)


class DiscreteVAE(nn.Module):
    def __init__(self, latent_dim, categorical_dim):
        super(DiscreteVAE, self).__init__()

        self.fc1 = nn.Linear(2352, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 2352)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.N = latent_dim
        self.K = categorical_dim

    def encoder(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decoder(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        out = self.sigmoid(self.fc6(h5))
        out = out.view(-1, 3, 28, 28)
        return out

    def forward(self, x, temp, hard):
        q = self.encoder(x.view(-1, 2352))
        q_y = q.view(q.size(0), self.N, self.K)
        z = gumbel_softmax(q_y, temp, hard)
        return self.decoder(z), F.softmax(q_y, dim=-1).reshape(q.size(0) * self.N, self.K)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]

    log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD


def train_discrete(num_epochs=30, temp=1.0, hard=False):
    temp_min = 0.5
    ANNEAL_RATE = 0.00003
    N = 3
    K = 20
    model = DiscreteVAE(N, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    train_loss = 0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")
        print("-" * 40)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, qy = model(x, temp, hard)
            loss = loss_function(x_hat, x, qy)
            loss.backward()
            train_loss += loss.item() * len(x)
            optimizer.step()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        print(f"Epochs loss: {train_loss} \n")
        train_loss = 0.0

    with open('discrete_vae.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model


def plot_digits_discrete(model):
    N = 3
    K = 20
    ind = torch.zeros(N, 1).long()
    images_list = []
    for k in range(K):
        to_generate = torch.zeros(K * K, N, K)
        index = 0
        for i in range(K):
            for j in range(K):
                ind[1] = k
                ind[0] = i
                ind[2] = j
                z = F.one_hot(ind, num_classes=K).squeeze(1)
                to_generate[index] = z
                index += 1

        generate = to_generate.view(-1, K * N).to(device)
        reconst_images = model.decoder(generate)
        reconst_images = reconst_images.view(reconst_images.size(0), 3, 28, 28).detach().cpu()
        grid_img = torchvision.utils.make_grid(reconst_images, nrow=K).permute(1, 2, 0).numpy() * 255
        grid_img = grid_img.astype(np.uint8)
        images_list.append(Image.fromarray(grid_img))

    for i in range(len(images_list)):
        plt.imshow(images_list[i])
        plt.show()


def main_discrete():
    model = train_discrete(num_epochs=30, temp=1.0, hard=False)
    plot_digits_discrete(model)


# Joint variational autoencoder
class JointVAE(nn.Module):
    def __init__(self, z_dim, N, K):
        super(JointVAE, self).__init__()
        # Encoding
        self.fc1 = nn.Linear(2352, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_to_d = nn.Linear(256, N * K)
        self.fc_to_n = nn.Linear(256, 2 * z_dim)

        # Decoding
        self.fc3 = nn.Linear(z_dim + N * K, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 2352)

        self.N = N
        self.K = K
        self.normal_latent_dim = z_dim

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def reparametrization_trick(self, mu, log_var):
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps

    def encoder_normal(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc_to_n(h2)
        mu, log_var = torch.split(h3, self.normal_latent_dim, dim=-1)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        z = self.reparametrization_trick(mu, log_var)
        return z, kl

    def encoder_discrete(self, x, temp, hard):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        q = self.relu(self.fc_to_d(h2))
        q_y = q.view(q.size(0), self.N, self.K)
        z = gumbel_softmax(q_y, temp, hard)
        return z, F.softmax(q_y, dim=-1).reshape(q.size(0) * self.N, self.K)

    def encoder_both(self, x, temp, hard):
        input = x.view(-1, 2352)
        z_normal, kl = self.encoder_normal(input)
        z_discrete, qy = self.encoder_discrete(input, temp, hard)
        output = torch.cat((z_normal, z_discrete), dim=1)
        return output, qy, kl

    def decoder(self, x):
        output = F.relu(self.fc3(x))
        output = F.relu(self.fc4(output))
        output = self.sigmoid(self.fc5(output))
        output = output.view(-1, 3, 28, 28)
        return output

    def forward(self, input, temp, hard):
        x, qy, kl = self.encoder_both(input, temp, hard)
        x = self.decoder(x)
        return x, qy, kl


def loss_function_joint(recon_x, x, qy, kl):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]

    log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD + kl


def train_joint(model, num_epochs=20, temp=1.0, hard=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    temp_min = 0.5
    ANNEAL_RATE = 0.00003
    model.train()
    train_loss = 0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")
        print("-" * 40)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, qy, kl = model(x, temp, hard)
            loss = loss_function_joint(x_hat, x, qy, kl)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        print(f"Epochs loss: {train_loss} \n")
        train_loss = 0.0

    with open('joint_vae.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model


def plot_digits_joint(model):
    N, K, z_dim = 3, 20, 2
    ind = torch.zeros(N, 1).long()
    images_list = []
    for k in range(K):
        to_generate = torch.zeros(K * K, N, K)
        index = 0
        for i in range(K):
            for j in range(K):
                ind[1] = k
                ind[0] = i
                ind[2] = j
                z = F.one_hot(ind, num_classes=K).squeeze(1)
                to_generate[index] = z
                index += 1

        generate = to_generate.view(-1, K * N).to(device)
        z_continuous = torch.normal(0, 3, size=(generate.shape[0], z_dim)).to(device)
        generate = torch.cat((z_continuous, generate), dim=1)
        reconst_images = model.decoder(generate)
        reconst_images = reconst_images.view(reconst_images.size(0), 3, 28, 28).detach().cpu()
        grid_img = torchvision.utils.make_grid(reconst_images, nrow=K).permute(1, 2, 0).numpy() * 255
        grid_img = grid_img.astype(np.uint8)
        images_list.append(Image.fromarray(grid_img))

    for i in range(len(images_list)):
        plt.imshow(images_list[i])
        plt.show()


def main_joint():
    z_dim = 2
    N = 3
    K = 20
    model = JointVAE(z_dim, N, K).to(device)
    model = train_joint(model, 30)
    plot_digits_joint(model)


if __name__ == '__main__':
    main_continuous()
    main_discrete()
    main_joint()
