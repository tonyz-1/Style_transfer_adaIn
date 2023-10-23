import argparse
from datetime import datetime
from glob import glob

import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, RandomSampler
from pathlib import Path
from tensorboardX import SummaryWriter
from torchsummary import torchsummary
from torchvision import datasets, transforms

import AdaIN_net


def train():
    print('training...')
    model.train()
    losses_train = []
    losses_c_train = []
    losses_s_train = []
    device = torch.device('cpu')

    for epoch in range(1, epochs + 1):
        print('epoch ', epoch)
        loss_c = 0.0
        loss_s = 0.0
        loss_c_train = 0.0
        loss_s_train = 0.0
        loss_train = 0.0
        totalBatches = int(len(content_set) / batchSize)
        for batch in range(1, totalBatches + 1):
            print('batch ', batch)
            content_imgs = next(content_iter).to(device)
            style_imgs = next(style_iter).to(device)

            loss_c, loss_s = model(content_imgs, style_imgs)
            loss_c = 1 * loss_c
            loss_s = 10 * loss_s
            loss = loss_c + loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train = loss.item()
            loss_c_train = loss_c.item()
            loss_s_train = loss_s.item()

        losses_c_train += [loss_c_train / totalBatches]
        losses_s_train += [loss_s_train / totalBatches]
        losses_train += [loss_train / totalBatches]

        scheduler.step()
        print('{} Epoch {}, Content loss {}, Style loss {}'.format(
            datetime.now(), epoch, loss_c_train / totalBatches, loss_s_train / totalBatches))

    xs = [x for x in range(len(losses_c_train))]
    plt.plot(xs, losses_c_train, label="Content Loss")

    xs = [x for x in range(len(losses_s_train))]
    plt.plot(xs, losses_s_train, label="Content Loss")

    xs = [x for x in range(len(losses_train))]
    plt.plot(xs, losses_train, label="Total Loss")

    plt.draw()

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(loss_plot)
    state_dict = model.decoder.state_dict()
    torch.save(state_dict, dec_weightPath)
    # torchsummary.summary(model(encoder, decoder), (1, 28 * 28))


class ImageDataset(Dataset):
    def __init__(self, root, transform):
        super(ImageDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.img_paths = list(Path(self.root).glob('*'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(str(img_path)).convert('RGB')
        img = self.transform(img)
        return img


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


parser = argparse.ArgumentParser()
parser.add_argument('-content_dir', '--content_dir', type=str)
parser.add_argument('-style_dir', '--style_dir', type=str)
parser.add_argument('-gamma', '--gamma', type=float)
parser.add_argument('-e', '--epoch', type=int)
parser.add_argument('-b', '--batchSize', type=int)
parser.add_argument('-l', '--encoder_weights')
parser.add_argument('-s', '--decoder_weights')
parser.add_argument('-p', '--loss_plot')
args = parser.parse_args()

content_dir = args.content_dir
style_dir = args.style_dir
gamma = args.gamma
epochs = args.epoch
batchSize = args.batchSize
enc_weightPath = args.encoder_weights
dec_weightPath = args.decoder_weights
loss_plot = args.loss_plot

encoder = AdaIN_net.encoder_decoder.encoder
encoder.load_state_dict(torch.load(enc_weightPath))
model = AdaIN_net.AdaIN_net(encoder)

content_set = ImageDataset(content_dir, train_transform())
style_set = ImageDataset(style_dir, train_transform())

content_loader = DataLoader(content_set, batchSize,
                            sampler=RandomSampler(content_set, replacement=True, num_samples=10 ** 100))
style_loader = DataLoader(style_set, batchSize,
                          sampler=RandomSampler(style_set, replacement=True, num_samples=10 ** 100))

content_iter = iter(content_loader)
style_iter = iter(style_loader)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = torch.nn.MSELoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

train()
