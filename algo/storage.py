import numpy as np
import torch
import ray
import random

class ReplayBuffer(object):
    def __init__(self, state_dim, N, H, gamma, gae_coeff):
        self.NH = N * H
        self.ptr_1, self.ptr_2 = 0, 0
        self.size = 0
        self.states = torch.zeros((self.NH, state_dim), dtype=torch.float32)
        self.next_states = torch.zeros((self.NH, state_dim), dtype=torch.float32)
        self.rewards = torch.zeros(self.NH, dtype=torch.float32)
        self.dones = torch.zeros(self.NH, dtype=torch.int32)
        self.alives = torch.zeros(self.NH, dtype=torch.int32)
        self.returns = torch.zeros(self.NH, dtype=torch.float32)
        self.values = torch.zeros(self.NH, dtype=torch.float32)
        self.next_values = torch.zeros(self.NH, dtype=torch.float32)
        self.noise_first = torch.zeros(self.NH, dtype=torch.int32)
        self.noise_idxes = torch.zeros(H, dtype=torch.int32)
        self.advantages = torch.zeros(H, dtype=torch.float32)
        self.last_ptr_1 = 0
        self.gamma = gamma
        self.gae_coeff = gae_coeff

    def add(self, batch_data):
        assert self.ptr_1 + batch_data['steps'] < self.NH + 1
        self.last_ptr_1 = self.ptr_1
        self.ptr_1 += batch_data['steps']
        self.states[self.last_ptr_1:self.ptr_1] = batch_data['states']
        self.next_states[self.last_ptr_1:self.ptr_1] = batch_data['next_states']
        self.rewards[self.last_ptr_1:self.ptr_1] = batch_data['rewards']
        self.dones[self.ptr_1-1] = batch_data['is_done']
        self.alives[self.last_ptr_1:self.ptr_1] = batch_data['not_deads']
        self.noise_first[self.last_ptr_1] = 1
        self.noise_idxes[self.ptr_2] = batch_data['noise_idx']
        self.ptr_2 += 1
    
    def get_transition(self):
        return (
            self.states[:self.ptr_1],
            self.next_states[:self.ptr_1],
            self.rewards[:self.ptr_1],
            self.dones[:self.ptr_1],
            self.alives[:self.ptr_1],
        )
    
    def store(self, values, next_values, returns, advantages):
        self.values[:self.ptr_1] = values
        self.next_values[:self.ptr_1] = next_values
        self.returns[:self.ptr_1] = returns
        self.advantages[:self.ptr_2] = advantages
    
    def get_item(self, name='states'):
        allowable_items = ['states', 'next_states', 'rewards', 'dones', 'alives', 'returns', 'values', 'next_values', 'returns']
        if name not in allowable_items:
            raise ValueError('name must be one of {}'.format(allowable_items))
        else:
            return getattr(self, name)[:self.ptr_1]

    def reset(self):
        self.ptr_1, self.ptr_2 = 0, 0
        self.last_ptr_1 = 0
        self.noise_first = torch.zeros(self.NH, dtype=torch.int32)