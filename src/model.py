import torch
import torch.nn as nn
import torch.autograd as autograd
import random
import numpy as np
import sys
import torch.nn.functional as F
sys.path.append("./auto_LiRPA")
from auto_LiRPA import BoundedModule
import math


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class QNetwork(nn.Module):
    def __init__(self, name, env, input_shape, num_actions, robust=False, width=1):
        super(QNetwork, self).__init__()
        self.env = env
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.robust = robust
        print(name)
        if name == 'DQN':
            self.features = nn.Sequential(
                nn.Linear(input_shape[0], 128*width),
#                nn.Linear(128, 128*width),
                nn.ReLU(),
                nn.Linear(128*width, 128*width),
                nn.ReLU(),
                nn.Linear(128*width, self.env.action_space.n)
            )
        elif name == 'CnnDQN':
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 32*width, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32*width, 64*width, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64*width, 64*width, kernel_size=3, stride=1),
                nn.ReLU(),
                Flatten(),
                nn.Linear(3136*width, 512*width),
                nn.ReLU(),
                nn.Linear(512*width, self.num_actions)
            )
        elif name == 'DuelingCnnDQN':
            self.features = DuelingCnnDQN(input_shape, num_actions, width)
        else:
            raise NotImplementedError('{} network structure not implemented.'.format(name))

        if self.robust:
            input_shape = input_shape[1:]
            dummy_input = torch.empty_like(torch.randn((1,) + input_shape))
            self.features = BoundedModule(self.features, dummy_input, device="cuda" if USE_CUDA else "cpu")

    def forward(self, *args, **kwargs):
        if len(args[0].shape) == 3:
            args = (args[0].unsqueeze(0),)
            #args[0] = args[0].unsqueeze(0)
        return self.features(*args, **kwargs)

    def act(self, state, epsilon=0):
        #state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        if self.robust:
            q_value = self.forward(state, method_opt='forward')
        else:
            q_value = self.forward(state)
        action  = q_value.max(1)[1]
        # mask = np.random.choice(np.arange(0, 2), p=[1-epsilon, epsilon])
        # action = (1-mask) * action + mask * np.random.randint(self.env.num_actions, size=state.size()[0])
        return action


class DuelingCnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, width=1):
        super(DuelingCnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32*width, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32*width, 64*width, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64*width, 64*width, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
        )

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 16*width, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(16*width, 32*width, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32*width, 16*width, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     Flatten(),
        # )

        self.advantage = nn.Sequential(
            nn.Linear(3136*width, 512*width),
            nn.ReLU(),
            nn.Linear(512*width, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(3136*width, 512*width),
            nn.ReLU(),
            nn.Linear(512*width, 1)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        cnn = self.cnn(x)
        advantage = self.advantage(cnn)
        value = self.value(cnn)
        return value + advantage - torch.sum(advantage, dim=1, keepdim=True) / self.num_actions


def model_setup(env_id, env, robust_model, logger, use_cuda, dueling=False, model_width=1):
    if "NoFrameskip" not in env_id and 'doom' not in env_id and 'airsim' not in env_id:
    # if False:
        net_name = 'DQN'
    else:
        if not dueling:
            net_name = 'CnnDQN'
        else:
            net_name = 'DuelingCnnDQN'
    #print(env.num_actions)
    model = QNetwork(net_name, env, env.observation_space.shape, env.num_actions, robust_model, model_width)
    if use_cuda:
        model = model.cuda()
    return model
