import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T, utils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import gym
import sys
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
from autoencoder_models import *
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

def exists(x):
    return x is not None

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.paths.sort(key=lambda p: self.extract_number_from_filename(p.name))

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def extract_number_from_filename(self, filename):
    # Extract the number from the filename using regex
        match = re.search(r'(\d+)', filename)
        return int(match.group()) if match else 0
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# def generatedataset(num_frames, env_id = "PongNoFrameskip-v4", max_diff_norm  1/255):
#
#     # env_params = {}
#     # env_params["crop_shift"] = 10
#     # env_params["restrict_actions"] = 4
#     # # env = gym.make(env_id)
#     # # env = make_env(env, frame_stack = False, scale = False)
#     # env = make_atari(env_id)
#     # env = wrap_deepmind(env, **env_params)
#     # env = wrap_pytorch(env)
#     env_params = {}
#     env_params["crop_shift"] = 10
#     env_params["restrict_actions"] = 4
#     env_id = "PongNoFrameskip-v4"
#     # env = gym.make(env_id)
#     # env = make_env(env, frame_stack = False, scale = False)
#     env = make_atari(env_id)
#     env = wrap_deepmind(env, **env_params)
#     env = wrap_pytorch(env)
#     encoder = model_setup(env_id, env, False, None, True, True, 1)
#     encoder.features.load_state_dict(torch.load("vanila_model.pth"))
#
#     all_obs = []
#     obs = env.reset()
#     obs = obs/255
#     count = 0
#     for i in range(num_frames):
#         count += 1
#         #action = env.action_space.sample()
#         obs_tensor = torch.from_numpy(np.ascontiguousarray(obs)).unsqueeze(0).cuda().to(torch.float32)
#         action = encoder.act(obs_tensor)[0]
#         obs ,_ , done , _ = env.step(action)
#         obs = obs/255
#         #print(obs)
#         all_obs.append(obs)
#         if done or count >= 2000:
#             count = 0
#             env.reset()
#     noisy_obs = []
#     for obs in all_obs:
#         for _ in range(10):
#             noisy = np.random.uniform(-max_diff_norm, max_diff_norm, size=(1,84,84))
#             noisy_image = np.clip(obs+noisy, 0, 1)
#             #print(noisy_image)
#             noisy_obs.append(noisy_image)
#     labels = [0 for i in range(len(noisy_obs))]
#     noisy_obs_tensors = torch.from_numpy(np.asarray(noisy_obs)).to(device=device, dtype=torch.float)
#     labels_tensors = torch.from_numpy(np.asarray(labels)).to(device=device, dtype=torch.float)
#     noisy_obs_encoded = encoder.features.cnn(noisy_obs_tensors).detach()
#     print(noisy_obs_encoded.shape)
#     print(labels_tensors.shape)
#     dataset = torch.utils.data.TensorDataset(noisy_obs_encoded, labels_tensors)
#     train_set, val_set = torch.utils.data.random_split(dataset, [int(0.95*len(dataset)), int(0.05*len(dataset))])
#     return train_set, val_set

class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64*7*7, zDim=512):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 32, kernel_size = 8, stride = 4)
        self.encConv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.encConv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 1)
        self.decConv2 = nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2)
        self.decConv3 = nn.ConvTranspose2d(32, imgChannels, kernel_size = 8, stride = 4)
        self.flatten = nn.Flatten(start_dim = 1)
        self.featureDim = featureDim
        self.zDim = zDim

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        #print(x.shape)
        x = F.relu(self.encConv1(x))
        #print(x.shape)
        x = F.relu(self.encConv2(x))
        #print(x.shape)
        x = F.relu(self.encConv3(x))
        #print(x.shape)
        x = x.view(-1, self.featureDim)
        #x = self.flatten(x)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        #x = torch.
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = torch.sigmoid(self.decConv3(x))
        ##x = self.decConv3(x)
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

class AE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=3136, zDim=128):
        super(AE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        # self.encConv1 = nn.Conv2d(imgChannels, 32, kernel_size = 8, stride = 4)
        # self.encConv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        # self.encConv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.encFC1 = nn.Linear(featureDim,256)
        self.encFC2 = nn.Linear(256, zDim)
        #self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, 256)
        self.decFC2 = nn.Linear(256, featureDim)
        # self.decConv1 = nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 1)
        # self.decConv2 = nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2)
        # self.decConv3 = nn.ConvTranspose2d(32, imgChannels, kernel_size = 8, stride = 4)
        self.flatten = nn.Flatten(start_dim = 1)
        self.featureDim = featureDim
        self.zDim = zDim

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        #print(x.shape)
        # x = F.relu(self.encConv1(x))
        # #print(x.shape)
        # x = F.relu(self.encConv2(x))
        # #print(x.shape)
        # x = F.relu(self.encConv3(x))
        #print(x.shape)
        #x = x.view(-1, self.featureDim)
        x = self.flatten(x)
        x = self.encFC1(x)
        x = self.encFC2(x)
        return x

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = F.relu(self.decFC2(x))
        #x = torch.
        # x = x.view(-1, 64, 7, 7)
        # x = F.relu(self.decConv1(x))
        # x = F.relu(self.decConv2(x))
        # x = torch.sigmoid(self.decConv3(x))
        ##x = self.decConv3(x)
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        z = self.encoder(x)
        out = self.decoder(z)
        return out

if __name__ == '__main__':

    #train_set, val_set = generatedataset(10000)
    image_size = 84
    dataset = Dataset('/freeway_pic_traj', image_size)
    short = range(0,50000)
    dataset = torch.utils.data.Subset(dataset, short)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.95*len(dataset)), int(0.05*len(dataset))])
    """
    Initialize Hyperparameters
    """
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 50
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size = 1)

    """
    Initialize the network and the Adam optimizer
    """
    net = Norm_3d_15_ae(16,ResidualBlock,84,64).to(device)
    #net = AE().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    loss_function = torch.nn.MSELoss()

    """
    Training the network for a given number of epochs
    The loss after every epoch is printed
    """
    for epoch in tqdm(range(num_epochs)):
        for idx, data in enumerate(train_loader, 0):
            #print(idx,data)
            imgs = data
            imgs = imgs.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            # out, mu, logVar = net(imgs)
            # out = net(imgs)
            # print("out",out[0])
            # print("true",imgs[0])

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            # kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            # print("kl ",kl_divergence)
            # loss1 = F.binary_cross_entropy(out.view(-1,84*84), imgs.view(-1,84*84), reduction = 'sum')/out.shape[0]
            # print("ce ",loss1)
            # loss = loss1 + kl_divergence

            out = net(imgs)
            loss = loss_function(out, imgs)
            #print(loss)
            #loss = F.binary_cross_entropy(out, imgs, size_average=False)


            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print('Epoch {}: Loss {}'.format(epoch, loss))
        #print(out[0].shape)
        cv2.imwrite('./test_pic/test_pic_'+str(epoch)+'.png', (out[0].detach().cpu().numpy().transpose(1,2,0))*255)
        torch.save(net, "./ae/pong_autoencoder"+str(epoch))

    # """
    # The following part takes a random image from test loader to feed into the VAE.
    # Both the original image and generated image from the distribution are shown.
    # """
    #
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import random
    #
    # torch.save(net, "pong_autoencoder")
    # net.eval()
    # with torch.no_grad():
    #     for data in random.sample(list(test_loader), 1):
    #         imgs, _ = data
    #         imgs = imgs.to(device)
    #         #print(imgs)
    #         out, mu, logVAR = net(imgs)
    #         # our = net(imgs)
    #         # print(imgs)
    #         # print(our)
    #         # print(imgs.shape)
    #         # print(our.shape)
    #         #break
