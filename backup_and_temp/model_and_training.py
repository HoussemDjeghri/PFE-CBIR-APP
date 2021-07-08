import time
import copy
import pickle
from barbar import Bar
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
# %matplotlib inline
from torch.optim import lr_scheduler  #this was commented
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchsummary import summary
from tqdm import tqdm
from pathlib import Path
import gc
from flask import Flask, render_template, request, redirect, url_for

RANDOMSTATE = 0
import os

app = Flask(__name__)


def preprocessing(img):
    x, y = img.size
    size = max(512, x, y)
    new_im = Image.new('RGB', (size, size), (0, 0, 0))
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


class CBIRDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame

        self.transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        row = self.dataFrame.iloc[key]
        img = Image.open(row['image'])
        new_img = preprocessing(img)
        image = self.transformations(new_img)
        return image

    def __len__(self):
        return len(self.dataFrame.index)


# Intermediate Function to process data from the data retrival class
def prepare_data(DF):
    trainDF, validateDF = train_test_split(DF,
                                           test_size=0.15,
                                           random_state=RANDOMSTATE)
    train_set = CBIRDataset(trainDF)
    validate_set = CBIRDataset(validateDF)

    return train_set, validate_set


class ConvAutoencoder_v2(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_v2, self).__init__()
        self.encoder = nn.Sequential(  # in- (N,3,512,512)
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1), nn.ReLU(True),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=1), nn.ReLU(True),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=0), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=1), nn.ReLU(True),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1), nn.ReLU(True),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=0),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=1),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=3,
                               kernel_size=(4, 4),
                               stride=2,
                               padding=2), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_ckpt(checkpoint_fpath, model, optimizer):

    # load check point
    checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['model_state_dict'])

    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # initialize valid_loss_min from checkpoint to valid_loss_min
    #valid_loss_min = checkpoint['valid_loss_min']

    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch']


def save_checkpoint(state, filename):
    """Save checkpoint if a new best is achieved"""
    print("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint


def train_model(
        model,
        criterion,
        optimizer,
        scheduler,  #this was commented
        num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for idx, inputs in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        # m = nn.Sigmoid()
                        # loss = nn.BCELoss()
                        # input = torch.randn(3, requires_grad=True)
                        # target = torch.empty(3).random_(2)
                        # output = loss(m(input), target)
                        # output.backward()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':  #this was commented
                scheduler.step()  #this was commented

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_checkpoint(state={
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                                filename='ckpt_epoch_{}.pt'.format(epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, optimizer, epoch_loss


if __name__ == '__main__':
    from multiprocessing import freeze_support

    # Find if any accelerator is presented, if yes switch device to use CUDA or else use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # preparing intermediate DataFrame
    datasetPath = Path('D:/Developement/PFE-APP/static/uploads/Dataset/')
    df = pd.DataFrame()

    myfilename = []
    for dirname, _, filenames in os.walk(datasetPath):
        for filename in filenames:
            #print(os.path.join(dirname, filename))
            myfilename.append(os.path.join(dirname, filename))
    df['image'] = myfilename

    #df['image'] = [f for f in os.listdir(datasetPath) if os.path.isfile(os.path.join(datasetPath, f))]
    #df['image'] = '/content/gdrive/MyDrive/Medical MNIST/test/' + df['image'].astype(str)
    df.head()

    EPOCHS = 3
    NUM_BATCHES = 16  #
    RETRAIN = False

    train_set, validate_set = prepare_data(DF=df)

    dataloaders = {
        'train':
        DataLoader(train_set,
                   batch_size=NUM_BATCHES,
                   shuffle=True,
                   num_workers=1),
        'val':
        DataLoader(validate_set, batch_size=NUM_BATCHES, num_workers=1)
    }

    #To show images
    # images = next(iter(DataLoader(train_set, batch_size=NUM_BATCHES, shuffle=True, num_workers=1)))
    # helper.imshow(images[31], normalize=False)

    dataset_sizes = {'train': len(train_set), 'val': len(validate_set)}

    model = ConvAutoencoder_v2().to(device)

    criterion = nn.MSELoss()
    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3,
                                           gamma=0.1)  #this was commented

    freeze_support()

    model, optimizer, loss = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=exp_lr_scheduler,  #this was commented
        num_epochs=EPOCHS)

    # Save the Trained Model
    torch.save(
        {
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        '/content/gdrive/MyDrive/Expérimentations documents/conv_autoencoder_v2_Exp#1.pt'
    )

    # extractor = parallelTestModule.ParallelExtractor()
    # extractor.runInParallel(numProcesses=2, numThreads=4)
#  python model_and_training.py