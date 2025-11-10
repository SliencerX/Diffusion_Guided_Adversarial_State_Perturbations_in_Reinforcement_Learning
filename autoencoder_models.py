import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Bernoulli, kl_divergence
from torch.autograd import Variable
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CropImage(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return crop_img_tensor(x, self.size)

def crop_img_tensor(x, size):
    """
    Crops a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple size.
    :param x: input image (tensor)
    :param size: iterable (height, width)
    :return: cropped image
    """
    return _pad_crop_img(x, size, 'crop')

def _pad_crop_img(x, size, mode):
    """
    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple size.
    :param x: input image (tensor)
    :param size: tuple (height, width)
    :param mode: string ('pad' | 'crop')
    :return: padded image
    """
    assert x.dim() == 4 and len(size) == 2
    size = tuple(size)
    x_size = x.size()[2:4]
    if mode == 'pad':
        cond = x_size[0] > size[0] or x_size[1] > size[1]
    elif mode == 'crop':
        cond = x_size[0] < size[0] or x_size[1] < size[1]
    else:
        raise ValueError("invalid mode '{}'".format(mode))
    if cond:
        raise ValueError('trying to {} from size {} to size {}'.format(
            mode, x_size, size))
    dr, dc = (abs(x_size[0] - size[0]), abs(x_size[1] - size[1]))
    dr1, dr2 = dr // 2, dr - (dr // 2)
    dc1, dc2 = dc // 2, dc - (dc // 2)
    if mode == 'pad':
        return nn.functional.pad(x, [dc1, dc2, dr1, dr2, 0, 0, 0, 0])
    elif mode == 'crop':
        return x[:, :, dr1:x_size[0] - dr2, dc1:x_size[1] - dc2]

def linear_schedule(alpha, epoch):
    schedule={
            0:0,
            (100//3):0,
            (100//3)*2:alpha
        }
    pkey = None
    pvalue = None
    for key, value in sorted(schedule.items(),reverse=True):
        # from large to small
        key = int(key) # for when restoring from the json file
        if key <= epoch:
            if pkey is None:
                return value
            else:
                return \
                    pvalue + \
                    ( epoch - pkey ) * ( value - pvalue ) / ( key - pkey )
        else:               # epoch < key
            pkey, pvalue = key, value
    return pvalue

def to_np(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return x

def is_conv(module: nn.Module) -> bool:
    """Returns whether the module is a convolutional layer."""
    return isinstance(module, torch.nn.modules.conv._ConvNd)

def is_linear(module: nn.Module) -> bool:
    """Returns whether the module is a linear layer."""
    return isinstance(module, torch.nn.Linear)

def _get_data_dep_hook(init_scale):
    """Creates forward hook for data-dependent initialization.
    The hook computes output statistics of the layer, corrects weights and
    bias, and corrects the output accordingly in-place, so the forward pass
    can continue.
    Args:
        init_scale (float): Desired scale (standard deviation) of each
            layer's output at initialization.
    Returns:
        Forward hook for data-dependent initialization
    """

    def hook(module, inp, out):
        inp = inp[0]

        out_size = out.size()

        if is_conv(module):
            separation_dim = 1
        elif is_linear(module):
            separation_dim = -1
        dims = tuple([i for i in range(out.dim()) if i != separation_dim])
        mean = out.mean(dims, keepdim=True)
        var = out.var(dims, keepdim=True)

        if True:
            print("Shapes:\n   input:  {}\n   output: {}\n   weight: {}".format(
                inp.size(), out_size, module.weight.size()))
            print("Dims to compute stats over:", dims)
            print("Input statistics:\n   mean: {}\n   var: {}".format(
                to_np(inp.mean(dims)), to_np(inp.var(dims))))
            print("Output statistics:\n   mean: {}\n   var: {}".format(
                to_np(out.mean(dims)), to_np(out.var(dims))))
            print("Weight statistics:   mean: {}   var: {}".format(
                to_np(module.weight.mean()), to_np(module.weight.var())))

        # Given channel y[i] we want to get
        #   y'[i] = (y[i]-mu[i]) * is/s[i]
        #         = (b[i]-mu[i]) * is/s[i] + sum_k (w[i, k] * is / s[i] * x[k])
        # where * is 2D convolution, k denotes input channels, mu[i] is the
        # sample mean of channel i, s[i] the sample variance, b[i] the current
        # bias, 'is' the initial scale, and w[i, k] the weight kernel for input
        # k and output i.
        # Therefore the correct bias and weights are:
        #   b'[i] = is * (b[i] - mu[i]) / s[i]
        #   w'[i, k] = w[i, k] * is / s[i]
        # And finally we can modify in place the output to get y'.

        scale = torch.sqrt(var + 1e-5)

        # Fix bias
        module.bias.data = ((module.bias.data - mean.flatten()) * init_scale /
                            scale.flatten())

        # Get correct dimension if transposed conv
        transp_conv = getattr(module, 'transposed', False)
        ch_out_dim = 1 if transp_conv else 0  # TODO handle groups

        # Fix weight
        size = tuple(-1 if i == ch_out_dim else 1 for i in range(out.dim()))
        weight_size = module.weight.size()
        module.weight.data *= init_scale / scale.view(size)
        assert module.weight.size() == weight_size

        # Fix output in-place so we can continue forward pass
        out.data -= mean
        out.data *= init_scale / scale

        assert out.size() == out_size

    return hook

def data_dependent_init(model: nn.Module,
                        model_input_dict: dict,
                        init_scale: Optional[float] = .1) -> None:
    """Performs data-dependent initialization on a model.
    Updates each layer's weights such that its outputs, computed on a batch
    of actual data, have mean 0 and the same standard deviation. See the code
    for more details.
    Args:
        model (torch.nn.Module):
        model_input_dict (dict): Dictionary of inputs to the model.
        init_scale (float, optional): Desired scale (standard deviation) of
            each layer's output at initialization. Default: 0.1.
    """

    hook_handles = []
    modules = filter(lambda m: is_conv(m) or is_linear(m), model.modules())
    for module in modules:
        # Init module parameters before forward pass
        nn.init.kaiming_normal_(module.weight.data)
        module.bias.data.zero_()

        # Forward hook: data-dependent initialization
        hook_handle = module.register_forward_hook(
            _get_data_dep_hook(init_scale))
        hook_handles.append(hook_handle)

    # Forward pass one minibatch
    model(**model_input_dict)  # dry-run

    # Remove forward hooks
    for hook_handle in hook_handles:
        hook_handle.remove()

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.2, leaky=0.01, downsample=None):
        super(ResidualBlock, self).__init__()
        # TODO: Maybe change order of res block from, conv->acti->batch->drop
        # to batch->acti->conv->drop
        self.leaky = leaky
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(channels, channels, kernel_size=3,padding=1),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(channels, channels, kernel_size=3,padding=1),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.block(x)
        x = F.leaky_relu(x,self.leaky)
        return x



class Norm_3d_15(nn.Module):
    def __init__(self, latent, block, image_size,latent_variable_size):
        super(Norm_3d_15, self).__init__()
        self.latent = latent
        dropout = 0.2
        leaky = 0.01
        self.image_size = image_size
        self.latent_variable_size = latent_variable_size
        channels = 64
        self.pz = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.resize = latent*10*10

        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, self.latent, kernel_size=3, padding=1, stride=2), #4x4
        )

        self.fc1 = nn.Linear(self.resize, latent_variable_size)
        self.fc2 = nn.Linear(self.resize, latent_variable_size)

        self.d1 = nn.Linear(latent_variable_size, self.resize)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent, channels, 3, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, channels, 4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, 1, 4, stride=2),  # 86x86
            CropImage((image_size, image_size)),
            nn.Sigmoid(),
        )

        self.leakyrelu = nn.LeakyReLU(leaky)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        x = self.encoder(x)
        #print(x.shape)
        x = x.view(-1, self.resize)
        #print(x.shape)
        return self.fc1(x), self.fc2(x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if device:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.latent, 10, 10)
        h1 = self.decoder(h1)
        return h1

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def sample_prior(self, n_imgs, **kwargs):
        z = self.pz.sample(torch.tensor([n_imgs, self.latent_variable_size])).to(device)
        z = z.view(n_imgs,self.latent_variable_size)
        mean = self.decode(z)
        return mean  # pxz.sample()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

class Norm_3d_15_ae(nn.Module):
    def __init__(self, latent, block, image_size,latent_variable_size):
        super(Norm_3d_15_ae, self).__init__()
        self.latent = latent
        dropout = 0.2
        leaky = 0.01
        self.image_size = image_size
        self.latent_variable_size = latent_variable_size
        channels = 64
        self.pz = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.resize = latent*10*10

        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, self.latent, kernel_size=3, padding=1, stride=2), #4x4
        )

        self.fc1 = nn.Linear(self.resize, latent_variable_size)

        self.d1 = nn.Linear(latent_variable_size, self.resize)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent, channels, 3, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, channels, 4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, 1, 4, stride=2),  # 86x86
            CropImage((image_size, image_size)),
            nn.Sigmoid(),
        )

        self.leakyrelu = nn.LeakyReLU(leaky)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        x = self.encoder(x)
        #print(x.shape)
        x = x.view(-1, self.resize)
        #print(x.shape)
        return self.fc1(x)

    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     if device:
    #         eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     else:
    #         eps = torch.FloatTensor(std.size()).normal_()
    #     eps = Variable(eps)
    #     return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.latent, 10, 10)
        h1 = self.decoder(h1)
        return h1

    # def get_latent_var(self, x):
    #     mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
    #     z = self.reparametrize(mu, logvar)
    #     return z

    def sample_prior(self, n_imgs, **kwargs):
        z = self.pz.sample(torch.tensor([n_imgs, self.latent_variable_size])).to(device)
        z = z.view(n_imgs,self.latent_variable_size)
        mean = self.decode(z)
        return mean  # pxz.sample()

    def forward(self, x):
        z = self.encode(x)
        #z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res

class Norm_3d_Conv_15(nn.Module):
    def __init__(self, latent, block, image_size):
        super(Norm_3d_Conv_15, self).__init__()
        self.latent = latent
        dropout = 0.2
        leaky = 0.01
        self.image_out_size = 15
        self.image_size = image_size
        channels = 64
        self.pz = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, self.latent, kernel_size=3, padding=1, stride=2), #4x4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d((int)(self.latent/2), channels, 3, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, channels, 4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, 1, 4, stride=2),  # 86x86
            CropImage((image_size, image_size)),
            nn.Sigmoid(),
        )

        self.leakyrelu = nn.LeakyReLU(leaky)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        x = self.encoder(x)
        mu,logvar = torch.chunk(x,2,dim=1)
        return mu,logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if device:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.decoder(z)
        return z

    def get_features(self, x):
        mu, logvar = self.encode(x)
        #z = self.reparametrize(mu, logvar)
        return mu.view(-1, (int)(self.latent/2) * self.image_out_size * self.image_out_size)

    def sample_prior(self, n_imgs, **kwargs):
        z = self.pz.sample(torch.tensor([n_imgs, (int)(self.latent/2),self.image_out_size, self.image_out_size])).to(device)
        z = z.view(n_imgs,(int)(self.latent/2),self.image_out_size, self.image_out_size)
        mean = self.decode(z)
        return mean  # pxz.sample()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar
