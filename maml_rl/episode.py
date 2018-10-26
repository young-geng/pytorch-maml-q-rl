import numpy as np
import torch
import torch.nn.functional as F

class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95, device='cpu'):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self._observations_list = [[] for _ in range(batch_size)]
        self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]
        self._mask_list = []

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None
        
        self._flattened_transitions = None
        
    @property
    def flattened_transitions(self):
        if self._flattened_transitions is None:
            observations = []
            next_observations = []
            actions = []
            dones = []
            rewards = []
            
            for i in range(self.batch_size):
                observations.append(np.stack(self._observations_list[i], axis=0))
                next_observations.append(
                    np.stack(
                        (self._observations_list[i][1:]
                         + [self._observations_list[i][0] * 0.0]),
                        axis=0
                    )
                )
                actions.append(
                    np.stack(self._actions_list[i], axis=0).astype(np.int64)
                )
                done = np.zeros(len(self._actions_list[i])).astype(np.float32)
                done[-1] = 1.0
                dones.append(done)
                rewards.append(self._rewards_list[i])
                
            observations = torch.from_numpy(
                np.concatenate(observations, axis=0)
            ).to(self.device)
            next_observations=  torch.from_numpy(
                np.concatenate(next_observations, axis=0)
            ).to(self.device)
            actions = torch.from_numpy(
                np.concatenate(actions, axis=0)
            ).to(self.device)
            dones = torch.from_numpy(
                np.concatenate(dones, axis=0)
            ).to(self.device)
            rewards = torch.from_numpy(
                np.concatenate(rewards, axis=0)
            ).to(self.device)
            
            self._flattened_transitions = (
                observations, actions, rewards, next_observations, dones 
            )
            
        return self._flattened_transitions

    @property
    def observations(self):
        if self._observations is None:
            observation_shape = self._observations_list[0][0].shape
            observations = np.zeros(
                (len(self), self.batch_size) + observation_shape,
                dtype=np.float32
            )
            for i in range(self.batch_size):
                length = len(self._observations_list[i])
                observations[:length, i] = np.stack(self._observations_list[i], axis=0)
            self._observations = torch.from_numpy(observations).to(self.device)
        return self._observations

    @property
    def actions(self):
        if self._actions is None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size)
                + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                actions[:length, i] = np.stack(self._actions_list[i], axis=0)
            self._actions = torch.from_numpy(actions).to(self.device)
        return self._actions

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._rewards_list[i])
                rewards[:length, i] = np.stack(self._rewards_list[i], axis=0)
            self._rewards = torch.from_numpy(rewards).to(self.device)
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards.cpu().numpy()
            mask = self.mask.cpu().numpy()
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + rewards[i] * mask[i]
                returns[i] = return_
            self._returns = torch.from_numpy(returns).to(self.device)
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                mask[:length, i] = 1.0
            self._mask = torch.from_numpy(mask).to(self.device)
        return self._mask

    def gae(self, values, tau=1.0):
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = values.squeeze(2).detach()
        values = F.pad(values * self.mask, (0, 0, 0, 1))

        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros_like(deltas[0]).float()
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * tau + deltas[i]
            advantages[i] = gae

        return advantages

    def append(self, observations, actions, rewards, batch_ids):
        for observation, action, reward, batch_id in zip(
                observations, actions, rewards, batch_ids):
            if batch_id is None:
                continue
            self._observations_list[batch_id].append(observation.astype(np.float32))
            self._actions_list[batch_id].append(action.astype(np.float32))
            self._rewards_list[batch_id].append(reward.astype(np.float32))

    def __len__(self):
        return max(map(len, self._rewards_list))
