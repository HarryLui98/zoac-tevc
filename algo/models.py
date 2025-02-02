import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import collections
import random
from scipy.linalg import toeplitz
import math

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"


def layer_init(in_size, out_size, nonlinear, init_method='default', actor_last=False):
    gain = nn.init.calculate_gain(nonlinear)
    if actor_last:
        gain *= 0.01
    module = nn.Linear(in_size, out_size)
    if init_method == 'orthogonal':
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0.)
    return module

class NeuralActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64, max_action=1, min_action=-1, seed=123):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        super().__init__()
        self.l1 = layer_init(input_dim, hidden_size, 'tanh', 'orthogonal')
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.l2 = layer_init(hidden_size, hidden_size, 'tanh', 'orthogonal')
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.l3 = layer_init(hidden_size, output_dim, 'tanh', 'orthogonal', actor_last=True)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_action = torch.Tensor(max_action)
        self.min_action = torch.Tensor(min_action)
        self.param_shapes = {
            k: tuple(self.state_dict()[k].size())
            for k in sorted(self.state_dict().keys())
        }
        self.para_num = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        a = torch.tanh(self.ln1(self.l1(x)))
        a = torch.tanh(self.ln2(self.l2(a)))
        a = (self.max_action - self.min_action) * torch.tanh(self.l3(a)) / 2 + (self.max_action + self.min_action) / 2
        return a

    def get_action(self, state, action_noise=False, action_noise_sigma=0.1):
        # net_input = torch.FloatTensor(state).to(device)
        action = self.__call__(state).detach().cpu().data.numpy().flatten()
        if not action_noise:
            return action
        noise = np.random.normal(0, 1, self.output_dim)
        return action + action_noise_sigma * noise

    def get_flat_param(self, after_map=True):
        theta_dict = self.state_dict()
        theta_list = []
        for k in sorted(theta_dict.keys()):
            theta_list.append(torch.reshape(theta_dict[k], (-1,)))
        flat_para = torch.cat(theta_list, dim=0).cpu().numpy()
        return flat_para

    def set_flat_param(self, theta, after_map=True):
        pos = 0
        theta_dict = self.state_dict()
        new_theta_dict = collections.OrderedDict()
        for k in sorted(theta_dict.keys()):
            shape = self.param_shapes[k]
            num_params = int(np.prod(shape))
            new_theta_dict[k] = torch.from_numpy(
                np.reshape(theta[pos: pos + num_params], shape)
            )
            pos += num_params
        self.load_state_dict(new_theta_dict)

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, cfg):
        seed = cfg['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        super().__init__()
        layers = []
        sizes = [input_dim] + [int(cfg['critic_hidden_size'])] * cfg['critic_hidden_layers'] + [output_dim]
        for j in range(len(sizes) - 1):
            if j < len(sizes) - 2:
                layers += [layer_init(sizes[j], sizes[j + 1], 'tanh', 'orthogonal'), nn.Tanh()]
            else:
                layers += [layer_init(sizes[j], sizes[j + 1], 'linear', 'orthogonal', actor_last=True)]
        self.value = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        v = self.value(x)
        return v
