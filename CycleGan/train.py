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
import math
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ColoredMNISTA(MNIST):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        super(ColoredMNISTA, self).__init__(root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.label_to_color_dict = {0: (156, 102, 31), 1: (0, 0, 255), 2: (127, 255, 0), 3: (255, 20, 147), 4: (255, 255, 0),
                    5: (255, 0, 0), 6: (128, 128, 128), 7: (186, 85, 211), 8: (255, 97, 3), 9: (0, 245, 255)}

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        img = img.numpy()
        color = self.label_to_color_dict[label.item()]
        img = np.stack((img, img, img), axis=2)
        img = np.where(img > 0, color, img)
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

class ColoredMNISTB(MNIST):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        super(ColoredMNISTB, self).__init__(root, train=train, download=download, transform=transform, target_transform=target_transform)

        self.label_to_color_dict = {9: (156, 102, 31), 0: (0, 0, 255), 1: (127, 255, 0), 2: (255, 20, 147), 3: (255, 255, 0),
                            4: (255, 0, 0), 5: (128, 128, 128), 6: (186, 85, 211), 7: (255, 97, 3), 8: (0, 245, 255)}

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        img = img.numpy()
        color = self.label_to_color_dict[label.item()]
        img = np.stack((img, img, img), axis=2)
        img = np.where(img > 0, color, img)
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

transform = transforms.Compose([transforms.ToTensor()])
colored_mnistA_train = ColoredMNISTA(root='./data', train=True, download=True, transform=transform)
colored_mnistA_test = ColoredMNISTA(root='./data', train=False, download=True, transform = transform)
train_loader_A = torch.utils.data.DataLoader(dataset=colored_mnistA_train, batch_size=120, num_workers = 4,pin_memory=True)

colored_mnistB_train = ColoredMNISTB(root='./data', train=True, download=True, transform=transform)
colored_mnistB_test = ColoredMNISTB(root='./data', train=False, download=True, transform = transform)
train_loader_B = torch.utils.data.DataLoader(dataset=colored_mnistB_train, batch_size=120, num_workers = 4, pin_memory=True)


class Dis_A(nn.Module):
    def __init__(self):
        super(Dis_A, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Dis_B(nn.Module):
    def __init__(self):
        super(Dis_B, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Gen_AB(nn.Module):
    def __init__(self):
        super(Gen_AB, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Gen_BA(nn.Module):
    def __init__(self):
        super(Gen_BA, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def generate_imgs(a, b, ab_gen, ba_gen):
    ab_gen.eval()
    ba_gen.eval()

    b_fake = ab_gen(a)
    a_fake = ba_gen(b)

    a_imgs = torch.zeros((a.shape[0] * 2, 3, a.shape[2], a.shape[3]))
    b_imgs = torch.zeros((b.shape[0] * 2, 3, b.shape[2], b.shape[3]))

    even_idx = torch.arange(start=0, end=a.shape[0] * 2, step=2)
    odd_idx = torch.arange(start=1, end=a.shape[0] * 2, step=2)

    a_imgs[even_idx] = a.cpu()
    a_imgs[odd_idx] = b_fake.cpu()

    b_imgs[even_idx] = b.cpu()
    b_imgs[odd_idx] = a_fake.cpu()

    rows = math.ceil((a.shape[0] * 2) ** 0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    a_imgs_ = torchvision.utils.make_grid(a_imgs, nrow=rows).permute(1, 2, 0).numpy() * 255
    a_imgs_ = a_imgs_.astype(np.uint8)
    ax1.imshow(Image.fromarray(a_imgs_))
    ax1.set_xticks([])
    ax1.set_yticks([])

    b_imgs_ = torchvision.utils.make_grid(b_imgs, nrow=rows).permute(1, 2, 0).numpy() * 255
    b_imgs_ = b_imgs_.astype(np.uint8)
    ax2.imshow(Image.fromarray(b_imgs_))
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()


def train(epochs, ab_gen, ba_gen, a_disc, b_disc):
    dis_a_opt = optim.Adam(a_disc.parameters(), lr=0.0003)
    dis_b_opt = optim.Adam(b_disc.parameters(), lr=0.0003)
    gen_ab_opt = optim.Adam(ab_gen.parameters(), lr=0.0003)
    gen_ba_opt = optim.Adam(ba_gen.parameters(), lr=0.0003)

    criterion = nn.BCELoss()

    a_fixed, _ = next(iter(train_loader_A))
    b_fixed, _ = next(iter(train_loader_B))
    a_fixed = a_fixed.to(device)
    b_fixed = b_fixed.to(device)

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1} / {epochs}")
        print("-" * 90)

        running_ab_gen_loss = 0.0
        running_ba_gen_loss = 0.0
        running_a_disc_loss = 0.0
        running_b_disc_loss = 0.0

        for i, (a_data, b_data) in enumerate(zip(train_loader_A, train_loader_B)):
            # Loading data
            a_real, _ = a_data
            b_real, _ = b_data

            a_real = a_real.to(device)
            b_real = b_real.to(device)

            batch_size = a_real.size(0)

            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)

            # ======================================= Training A Discriminator =======================================
            a_disc.train()
            ba_gen.train(False)

            ## real images
            a_real_out = a_disc(a_real)
            a_disc_real_loss = criterion(a_real_out.squeeze(1), real_labels)

            ## fake images
            a_fake = ba_gen(b_real)
            a_fake_out = a_disc(a_fake)
            a_disc_fake_loss = criterion(a_fake_out.squeeze(1), fake_labels)

            a_disc_total_loss = a_disc_real_loss + a_disc_fake_loss
            a_disc.zero_grad()
            running_a_disc_loss += a_disc_total_loss.item()
            a_disc_total_loss.backward()
            dis_a_opt.step()

            # ======================================= Training B -> A Generator =======================================
            a_disc.train(False)
            ba_gen.train()

            a_fake = ba_gen(b_real)
            a_fake_out = a_disc(a_fake)
            ba_gen_loss = criterion(a_fake_out.squeeze(1), real_labels)
            running_ba_gen_loss += ba_gen_loss.item()

            a_disc.zero_grad()
            ba_gen.zero_grad()
            ba_gen_loss.backward()
            gen_ba_opt.step()

            ba_gen.train(False)
            # ======================================= Training B Discriminator =======================================
            b_disc.train()
            ab_gen.train(False)

            ## real images
            b_real_out = b_disc(b_real)
            b_disc_real_loss = criterion(b_real_out.squeeze(1), real_labels)

            ## fake images
            b_fake = ab_gen(a_real)
            b_fake_out = b_disc(b_fake)
            b_disc_fake_loss = criterion(b_fake_out.squeeze(1), fake_labels)

            b_disc_total_loss = b_disc_real_loss + b_disc_fake_loss
            b_disc.zero_grad()
            running_b_disc_loss += b_disc_total_loss.item()
            b_disc_total_loss.backward()
            dis_b_opt.step()

            # ======================================= Training A -> B Generator =======================================
            b_disc.train(False)
            ab_gen.train()

            b_fake = ab_gen(a_real)
            b_fake_out = b_disc(b_fake)
            ab_gen_loss = criterion(b_fake_out.squeeze(1), real_labels)
            running_ab_gen_loss += ab_gen_loss.item()

            b_disc.zero_grad()
            ab_gen.zero_grad()
            ab_gen_loss.backward()
            gen_ab_opt.step()

            ab_gen.train(False)

        print(f"avg discriminator A loss: {round(running_a_disc_loss / len(train_loader_A), 4)}")
        print(f"avg discriminator B loss: {round(running_b_disc_loss / len(train_loader_A), 4)}")
        print(f"avg generator A->B loss: {round(running_ab_gen_loss / len(train_loader_A), 4)}")
        print(f"avg generator B->A loss: {round(running_ba_gen_loss / len(train_loader_A), 4)}")

        print("-" * 90)
        if epoch % 50 == 0:
            generate_imgs(a_fixed, b_fixed, ab_gen, ba_gen)

    generate_imgs(a_fixed, b_fixed, ab_gen, ba_gen)
    return ab_gen, ba_gen, a_disc, b_disc


def main():
    ab_gen = Gen_AB().to(device)
    ba_gen = Gen_BA().to(device)
    a_disc = Dis_A().to(device)
    b_disc = Dis_B().to(device)
    ab_gen, ba_gen, a_disc, b_disc = train(150, ab_gen, ba_gen, a_disc, b_disc)

    with open("ab_gen.pkl", "wb") as f:
        pickle.dump(ab_gen, f)
    with open("ba_gen.pkl", "wb") as f:
        pickle.dump(ba_gen, f)
    with open("a_disc.pkl", "wb") as f:
        pickle.dump(a_disc, f)
    with open("b_disc.pkl", "wb") as f:
        pickle.dump(b_disc, f)